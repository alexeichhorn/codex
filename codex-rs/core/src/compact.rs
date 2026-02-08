use std::sync::Arc;

use crate::ModelProviderInfo;
use crate::Prompt;
use crate::client::ModelClientSession;
use crate::client_common::ResponseEvent;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::codex::get_last_assistant_message_from_turn;
use crate::context_manager::ContextManager;
use crate::context_manager::estimate_item_token_count;
use crate::error::CodexErr;
use crate::error::Result as CodexResult;
use crate::features::Feature;
use crate::instructions::SkillInstructions;
use crate::instructions::UserInstructions;
use crate::protocol::CompactedItem;
use crate::protocol::EventMsg;
use crate::protocol::TurnContextItem;
use crate::protocol::TurnStartedEvent;
use crate::protocol::WarningEvent;
use crate::session_prefix::TURN_ABORTED_OPEN_TAG;
use crate::session_prefix::is_session_prefix;
use crate::truncate::TruncationPolicy;
use crate::truncate::approx_token_count;
use crate::truncate::truncate_text;
use crate::util::backoff;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::user_input::UserInput;
use futures::prelude::*;
use tracing::error;

pub const SUMMARIZATION_PROMPT: &str = include_str!("../templates/compact/prompt.md");
pub const SUMMARY_PREFIX: &str = include_str!("../templates/compact/summary_prefix.md");
const COMPACT_USER_MESSAGE_MAX_TOKENS: usize = 20_000;

/// Fraction of history (by tokens) to compact during auto-compaction.
/// The remaining (1 - fraction) is kept verbatim.
pub(crate) const COMPACT_SPLIT_FRACTION: f64 = 0.5;

/// Minimum token count in the old half before compaction is worthwhile.
const COMPACT_MIN_OLD_HALF_TOKENS: i64 = 1_000;

pub(crate) fn should_use_remote_compact_task(
    session: &Session,
    provider: &ModelProviderInfo,
) -> bool {
    provider.is_openai() && session.enabled(Feature::RemoteCompaction)
}

pub(crate) async fn run_inline_auto_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
) {
    let prompt = turn_context.compact_prompt().to_string();
    let input = vec![UserInput::Text {
        text: prompt,
        // Compaction prompt is synthesized; no UI element ranges to preserve.
        text_elements: Vec::new(),
    }];

    if sess.enabled(Feature::PartialCompaction) {
        run_partial_compact_task_inner(sess, turn_context, input).await;
    } else {
        run_compact_task_inner(sess, turn_context, input).await;
    }
}

/// Partial compaction: only summarize the older half of history,
/// keeping the recent half verbatim.
async fn run_partial_compact_task_inner(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: Vec<UserInput>,
) {
    let full_history = sess.clone_history().await;
    let all_items = full_history.raw_items().to_vec();

    // Try to split; if old half is too small, fall back to full compaction.
    let Some((old_half, recent_half)) =
        split_history_at_token_midpoint(&all_items, COMPACT_SPLIT_FRACTION)
    else {
        run_compact_task_inner(sess, turn_context, input).await;
        return;
    };

    let compaction_item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(&turn_context, &compaction_item)
        .await;

    // Build a temporary history containing only the old half + summarization prompt.
    let initial_input_for_turn: ResponseInputItem = ResponseInputItem::from(input);
    let mut compact_history = ContextManager::new();
    compact_history.replace(old_half.clone());
    compact_history.record_items(
        &[initial_input_for_turn.into()],
        turn_context.truncation_policy,
    );

    let mut truncated_count = 0usize;
    let max_retries = turn_context.provider.stream_max_retries();
    let mut retries = 0;
    let turn_metadata_header = turn_context.resolve_turn_metadata_header().await;
    let mut client_session = sess.services.model_client.new_session();

    let collaboration_mode = sess.current_collaboration_mode().await;
    let rollout_item = RolloutItem::TurnContext(TurnContextItem {
        cwd: turn_context.cwd.clone(),
        approval_policy: turn_context.approval_policy,
        sandbox_policy: turn_context.sandbox_policy.clone(),
        model: turn_context.model_info.slug.clone(),
        personality: turn_context.personality,
        collaboration_mode: Some(collaboration_mode),
        effort: turn_context.reasoning_effort,
        summary: turn_context.reasoning_summary,
        user_instructions: turn_context.user_instructions.clone(),
        developer_instructions: turn_context.developer_instructions.clone(),
        final_output_json_schema: turn_context.final_output_json_schema.clone(),
        truncation_policy: Some(turn_context.truncation_policy.into()),
    });
    sess.persist_rollout_items(&[rollout_item]).await;

    loop {
        let turn_input = compact_history.clone().for_prompt();
        let turn_input_len = turn_input.len();
        let prompt = Prompt {
            input: turn_input,
            base_instructions: sess.get_base_instructions().await,
            personality: turn_context.personality,
            ..Default::default()
        };
        let attempt_result = drain_to_completed(
            &sess,
            turn_context.as_ref(),
            &mut client_session,
            turn_metadata_header.as_deref(),
            &prompt,
        )
        .await;

        match attempt_result {
            Ok(()) => {
                if truncated_count > 0 {
                    sess.notify_background_event(
                        turn_context.as_ref(),
                        format!(
                            "Trimmed {truncated_count} older thread item(s) before compacting so the prompt fits the model context window."
                        ),
                    )
                    .await;
                }
                break;
            }
            Err(CodexErr::Interrupted) => {
                return;
            }
            Err(e @ CodexErr::ContextWindowExceeded) => {
                if turn_input_len > 1 {
                    error!(
                        "Context window exceeded while compacting; removing oldest history item. Error: {e}"
                    );
                    compact_history.remove_first_item();
                    truncated_count += 1;
                    retries = 0;
                    continue;
                }
                sess.set_total_tokens_full(turn_context.as_ref()).await;
                let event = EventMsg::Error(e.to_error_event(None));
                sess.send_event(&turn_context, event).await;
                return;
            }
            Err(e) => {
                if retries < max_retries {
                    retries += 1;
                    let delay = backoff(retries);
                    sess.notify_stream_error(
                        turn_context.as_ref(),
                        format!("Reconnecting... {retries}/{max_retries}"),
                        e,
                    )
                    .await;
                    tokio::time::sleep(delay).await;
                    continue;
                } else {
                    let event = EventMsg::Error(e.to_error_event(None));
                    sess.send_event(&turn_context, event).await;
                    return;
                }
            }
        }
    }

    // The model's summary is recorded in the session history via drain_to_completed.
    // Extract it from there.
    let history_snapshot = sess.clone_history().await;
    let history_items = history_snapshot.raw_items();
    let summary_suffix = get_last_assistant_message_from_turn(history_items).unwrap_or_default();
    let summary_text = format!("{SUMMARY_PREFIX}\n{summary_suffix}");

    let initial_context = sess.build_initial_context(turn_context.as_ref()).await;

    // Collect ghost snapshots from the old half (recent half's ghosts are already in recent_half).
    let ghost_snapshots: Vec<ResponseItem> = old_half
        .iter()
        .filter(|item| matches!(item, ResponseItem::GhostSnapshot { .. }))
        .cloned()
        .collect();

    let mut new_history =
        build_partial_compacted_history(initial_context, &summary_text, recent_half);
    new_history.extend(ghost_snapshots);

    sess.replace_history(new_history).await;
    sess.recompute_token_usage(&turn_context).await;

    let rollout_item = RolloutItem::Compacted(CompactedItem {
        message: summary_text.clone(),
        replacement_history: None,
    });
    sess.persist_rollout_items(&[rollout_item]).await;

    sess.emit_turn_item_completed(&turn_context, compaction_item)
        .await;
    let warning = EventMsg::Warning(WarningEvent {
        message: "Heads up: Long threads and multiple compactions can cause the model to be less accurate. Start a new thread when possible to keep threads small and targeted.".to_string(),
    });
    sess.send_event(&turn_context, warning).await;
}

pub(crate) async fn run_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: Vec<UserInput>,
) {
    let start_event = EventMsg::TurnStarted(TurnStartedEvent {
        model_context_window: turn_context.model_context_window(),
        collaboration_mode_kind: turn_context.collaboration_mode.mode,
    });
    sess.send_event(&turn_context, start_event).await;
    run_compact_task_inner(sess.clone(), turn_context, input).await;
}

async fn run_compact_task_inner(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
    input: Vec<UserInput>,
) {
    let compaction_item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(&turn_context, &compaction_item)
        .await;
    let initial_input_for_turn: ResponseInputItem = ResponseInputItem::from(input);

    let mut history = sess.clone_history().await;
    history.record_items(
        &[initial_input_for_turn.into()],
        turn_context.truncation_policy,
    );

    let mut truncated_count = 0usize;

    let max_retries = turn_context.provider.stream_max_retries();
    let mut retries = 0;
    let turn_metadata_header = turn_context.resolve_turn_metadata_header().await;
    let mut client_session = sess.services.model_client.new_session();
    // Reuse one client session so turn-scoped state (sticky routing, websocket append tracking)
    // survives retries within this compact turn.

    // TODO: If we need to guarantee the persisted mode always matches the prompt used for this
    // turn, capture it in TurnContext at creation time. Using SessionConfiguration here avoids
    // duplicating model settings on TurnContext, but an Op after turn start could update the
    // session config before this write occurs.
    let collaboration_mode = sess.current_collaboration_mode().await;
    let rollout_item = RolloutItem::TurnContext(TurnContextItem {
        cwd: turn_context.cwd.clone(),
        approval_policy: turn_context.approval_policy,
        sandbox_policy: turn_context.sandbox_policy.clone(),
        model: turn_context.model_info.slug.clone(),
        personality: turn_context.personality,
        collaboration_mode: Some(collaboration_mode),
        effort: turn_context.reasoning_effort,
        summary: turn_context.reasoning_summary,
        user_instructions: turn_context.user_instructions.clone(),
        developer_instructions: turn_context.developer_instructions.clone(),
        final_output_json_schema: turn_context.final_output_json_schema.clone(),
        truncation_policy: Some(turn_context.truncation_policy.into()),
    });
    sess.persist_rollout_items(&[rollout_item]).await;

    loop {
        // Clone is required because of the loop
        let turn_input = history.clone().for_prompt();
        let turn_input_len = turn_input.len();
        let prompt = Prompt {
            input: turn_input,
            base_instructions: sess.get_base_instructions().await,
            personality: turn_context.personality,
            ..Default::default()
        };
        let attempt_result = drain_to_completed(
            &sess,
            turn_context.as_ref(),
            &mut client_session,
            turn_metadata_header.as_deref(),
            &prompt,
        )
        .await;

        match attempt_result {
            Ok(()) => {
                if truncated_count > 0 {
                    sess.notify_background_event(
                        turn_context.as_ref(),
                        format!(
                            "Trimmed {truncated_count} older thread item(s) before compacting so the prompt fits the model context window."
                        ),
                    )
                    .await;
                }
                break;
            }
            Err(CodexErr::Interrupted) => {
                return;
            }
            Err(e @ CodexErr::ContextWindowExceeded) => {
                if turn_input_len > 1 {
                    // Trim from the beginning to preserve cache (prefix-based) and keep recent messages intact.
                    error!(
                        "Context window exceeded while compacting; removing oldest history item. Error: {e}"
                    );
                    history.remove_first_item();
                    truncated_count += 1;
                    retries = 0;
                    continue;
                }
                sess.set_total_tokens_full(turn_context.as_ref()).await;
                let event = EventMsg::Error(e.to_error_event(None));
                sess.send_event(&turn_context, event).await;
                return;
            }
            Err(e) => {
                if retries < max_retries {
                    retries += 1;
                    let delay = backoff(retries);
                    sess.notify_stream_error(
                        turn_context.as_ref(),
                        format!("Reconnecting... {retries}/{max_retries}"),
                        e,
                    )
                    .await;
                    tokio::time::sleep(delay).await;
                    continue;
                } else {
                    let event = EventMsg::Error(e.to_error_event(None));
                    sess.send_event(&turn_context, event).await;
                    return;
                }
            }
        }
    }

    let history_snapshot = sess.clone_history().await;
    let history_items = history_snapshot.raw_items();
    let summary_suffix = get_last_assistant_message_from_turn(history_items).unwrap_or_default();
    let summary_text = format!("{SUMMARY_PREFIX}\n{summary_suffix}");
    let user_messages = collect_user_messages(history_items);

    let initial_context = sess.build_initial_context(turn_context.as_ref()).await;
    let mut new_history = build_compacted_history(initial_context, &user_messages, &summary_text);
    let ghost_snapshots: Vec<ResponseItem> = history_items
        .iter()
        .filter(|item| matches!(item, ResponseItem::GhostSnapshot { .. }))
        .cloned()
        .collect();
    new_history.extend(ghost_snapshots);
    sess.replace_history(new_history).await;
    sess.recompute_token_usage(&turn_context).await;

    let rollout_item = RolloutItem::Compacted(CompactedItem {
        message: summary_text.clone(),
        replacement_history: None,
    });
    sess.persist_rollout_items(&[rollout_item]).await;

    sess.emit_turn_item_completed(&turn_context, compaction_item)
        .await;
    let warning = EventMsg::Warning(WarningEvent {
        message: "Heads up: Long threads and multiple compactions can cause the model to be less accurate. Start a new thread when possible to keep threads small and targeted.".to_string(),
    });
    sess.send_event(&turn_context, warning).await;
}

pub fn content_items_to_text(content: &[ContentItem]) -> Option<String> {
    let mut pieces = Vec::new();
    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                if !text.is_empty() {
                    pieces.push(text.as_str());
                }
            }
            ContentItem::InputImage { .. } => {}
        }
    }
    if pieces.is_empty() {
        None
    } else {
        Some(pieces.join("\n"))
    }
}

pub(crate) fn collect_user_messages(items: &[ResponseItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match crate::event_mapping::parse_turn_item(item) {
            Some(TurnItem::UserMessage(user)) => {
                if is_summary_message(&user.message()) {
                    None
                } else {
                    Some(user.message())
                }
            }
            _ => collect_turn_aborted_marker(item),
        })
        .collect()
}

fn collect_turn_aborted_marker(item: &ResponseItem) -> Option<String> {
    let ResponseItem::Message { role, content, .. } = item else {
        return None;
    };
    if role != "user" {
        return None;
    }

    let text = content_items_to_text(content)?;
    if text
        .trim_start()
        .to_ascii_lowercase()
        .starts_with(TURN_ABORTED_OPEN_TAG)
    {
        Some(text)
    } else {
        None
    }
}

pub(crate) fn is_summary_message(message: &str) -> bool {
    message.starts_with(format!("{SUMMARY_PREFIX}\n").as_str())
}

/// Returns true if the item is a call that expects a corresponding output.
fn is_call_item(item: &ResponseItem) -> bool {
    matches!(
        item,
        ResponseItem::FunctionCall { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::LocalShellCall { .. }
    )
}

/// Returns true if the item is an output that corresponds to a call.
fn is_output_item(item: &ResponseItem) -> bool {
    matches!(
        item,
        ResponseItem::FunctionCallOutput { .. } | ResponseItem::CustomToolCallOutput { .. }
    )
}

/// Returns true for history scaffolding that is re-injected via `build_initial_context`
/// and should not influence partial-compaction midpoint selection.
fn is_reinjected_scaffolding_item(item: &ResponseItem) -> bool {
    let ResponseItem::Message { role, content, .. } = item else {
        return false;
    };

    if role == "developer" {
        return true;
    }

    if role != "user" {
        return false;
    }

    if UserInstructions::is_user_instructions(content)
        || SkillInstructions::is_skill_instructions(content)
    {
        return true;
    }

    content.iter().any(|entry| match entry {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
            is_session_prefix(text)
        }
        ContentItem::InputImage { .. } => false,
    })
}

/// Split history items into (old_half, recent_half) by cumulative token count.
///
/// The split targets `split_fraction` of total tokens in the old half.
/// The boundary is adjusted so call/output pairs are never split across halves.
///
/// Returns `None` if the old half would have fewer than
/// `COMPACT_MIN_OLD_HALF_TOKENS` tokens (not worth compacting).
pub(crate) fn split_history_at_token_midpoint(
    items: &[ResponseItem],
    split_fraction: f64,
) -> Option<(Vec<ResponseItem>, Vec<ResponseItem>)> {
    if items.is_empty() {
        return None;
    }

    let split_eligible_indices: Vec<usize> = items
        .iter()
        .enumerate()
        .filter_map(|(idx, item)| (!is_reinjected_scaffolding_item(item)).then_some(idx))
        .collect();
    if split_eligible_indices.is_empty() {
        return None;
    }

    let total_tokens: i64 = split_eligible_indices
        .iter()
        .map(|idx| estimate_item_token_count(&items[*idx]))
        .sum();
    let target_tokens = (total_tokens as f64 * split_fraction) as i64;

    if target_tokens < COMPACT_MIN_OLD_HALF_TOKENS {
        return None;
    }

    // Walk forward to find the candidate split index.
    let mut accumulated: i64 = 0;
    let mut split_idx = items.len(); // default: everything in old half
    for idx in split_eligible_indices {
        accumulated += estimate_item_token_count(&items[idx]);
        if accumulated >= target_tokens {
            split_idx = idx + 1; // split after this item
            break;
        }
    }

    // Adjust for call/output pair boundaries.
    // Move all consecutive trailing calls from old_half into recent_half
    // (handles parallel tool calls like [Call_A, Call_B] that would be
    // orphaned from their outputs).
    while split_idx > 0 && is_call_item(&items[split_idx - 1]) {
        split_idx -= 1;
    }
    // Move all consecutive leading outputs from recent_half into old_half
    // (handles parallel outputs like [Output_A, Output_B] that would be
    // orphaned from their calls).
    while split_idx < items.len() && is_output_item(&items[split_idx]) {
        split_idx += 1;
    }

    // Clamp to valid range.
    split_idx = split_idx.clamp(0, items.len());

    // Re-check old half is still meaningful.
    let old_half_tokens: i64 = items[..split_idx]
        .iter()
        .filter(|item| !is_reinjected_scaffolding_item(item))
        .map(|i| estimate_item_token_count(i))
        .sum();
    if old_half_tokens < COMPACT_MIN_OLD_HALF_TOKENS {
        return None;
    }

    // Don't compact if there's nothing left in the recent half.
    if split_idx >= items.len() {
        return None;
    }

    Some((items[..split_idx].to_vec(), items[split_idx..].to_vec()))
}

/// Build a new history for partial compaction:
/// initial_context + summary_as_user_message + recent_half (verbatim).
fn build_partial_compacted_history(
    mut history: Vec<ResponseItem>,
    summary_text: &str,
    recent_half: Vec<ResponseItem>,
) -> Vec<ResponseItem> {
    let summary_text = if summary_text.is_empty() {
        "(no summary available)".to_string()
    } else {
        summary_text.to_string()
    };

    history.push(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText { text: summary_text }],
        end_turn: None,
        phase: None,
    });

    // Append the recent half verbatim — all item types preserved.
    history.extend(recent_half);

    history
}

pub(crate) fn build_compacted_history(
    initial_context: Vec<ResponseItem>,
    user_messages: &[String],
    summary_text: &str,
) -> Vec<ResponseItem> {
    build_compacted_history_with_limit(
        initial_context,
        user_messages,
        summary_text,
        COMPACT_USER_MESSAGE_MAX_TOKENS,
    )
}

fn build_compacted_history_with_limit(
    mut history: Vec<ResponseItem>,
    user_messages: &[String],
    summary_text: &str,
    max_tokens: usize,
) -> Vec<ResponseItem> {
    let mut selected_messages: Vec<String> = Vec::new();
    if max_tokens > 0 {
        let mut remaining = max_tokens;
        for message in user_messages.iter().rev() {
            if remaining == 0 {
                break;
            }
            let tokens = approx_token_count(message);
            if tokens <= remaining {
                selected_messages.push(message.clone());
                remaining = remaining.saturating_sub(tokens);
            } else {
                let truncated = truncate_text(message, TruncationPolicy::Tokens(remaining));
                selected_messages.push(truncated);
                break;
            }
        }
        selected_messages.reverse();
    }

    for message in &selected_messages {
        history.push(ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: message.clone(),
            }],
            end_turn: None,
            phase: None,
        });
    }

    let summary_text = if summary_text.is_empty() {
        "(no summary available)".to_string()
    } else {
        summary_text.to_string()
    };

    history.push(ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText { text: summary_text }],
        end_turn: None,
        phase: None,
    });

    history
}

async fn drain_to_completed(
    sess: &Session,
    turn_context: &TurnContext,
    client_session: &mut ModelClientSession,
    turn_metadata_header: Option<&str>,
    prompt: &Prompt,
) -> CodexResult<()> {
    let mut stream = client_session
        .stream(
            prompt,
            &turn_context.model_info,
            &turn_context.otel_manager,
            turn_context.reasoning_effort,
            turn_context.reasoning_summary,
            turn_metadata_header,
        )
        .await?;
    loop {
        let maybe_event = stream.next().await;
        let Some(event) = maybe_event else {
            return Err(CodexErr::Stream(
                "stream closed before response.completed".into(),
                None,
            ));
        };
        match event {
            Ok(ResponseEvent::OutputItemDone(item)) => {
                sess.record_into_history(std::slice::from_ref(&item), turn_context)
                    .await;
            }
            Ok(ResponseEvent::ServerReasoningIncluded(included)) => {
                sess.set_server_reasoning_included(included).await;
            }
            Ok(ResponseEvent::RateLimits(snapshot)) => {
                sess.update_rate_limits(turn_context, snapshot).await;
            }
            Ok(ResponseEvent::Completed { token_usage, .. }) => {
                sess.update_token_usage_info(turn_context, token_usage.as_ref())
                    .await;
                return Ok(());
            }
            Ok(_) => continue,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::session_prefix::TURN_ABORTED_OPEN_TAG;
    use pretty_assertions::assert_eq;

    #[test]
    fn content_items_to_text_joins_non_empty_segments() {
        let items = vec![
            ContentItem::InputText {
                text: "hello".to_string(),
            },
            ContentItem::OutputText {
                text: String::new(),
            },
            ContentItem::OutputText {
                text: "world".to_string(),
            },
        ];

        let joined = content_items_to_text(&items);

        assert_eq!(Some("hello\nworld".to_string()), joined);
    }

    #[test]
    fn content_items_to_text_ignores_image_only_content() {
        let items = vec![ContentItem::InputImage {
            image_url: "file://image.png".to_string(),
        }];

        let joined = content_items_to_text(&items);

        assert_eq!(None, joined);
    }

    #[test]
    fn collect_user_messages_extracts_user_text_only() {
        let items = vec![
            ResponseItem::Message {
                id: Some("assistant".to_string()),
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "ignored".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Message {
                id: Some("user".to_string()),
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "first".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Other,
        ];

        let collected = collect_user_messages(&items);

        assert_eq!(vec!["first".to_string()], collected);
    }

    #[test]
    fn collect_user_messages_filters_session_prefix_entries() {
        let items = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "# AGENTS.md instructions for project\n\n<INSTRUCTIONS>\ndo things\n</INSTRUCTIONS>"
                        .to_string(),
                }],
                end_turn: None,
            phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "<ENVIRONMENT_CONTEXT>cwd=/tmp</ENVIRONMENT_CONTEXT>".to_string(),
                }],
                end_turn: None,
            phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "real user message".to_string(),
                }],
                end_turn: None,
            phase: None,
            },
        ];

        let collected = collect_user_messages(&items);

        assert_eq!(vec!["real user message".to_string()], collected);
    }

    #[test]
    fn build_token_limited_compacted_history_truncates_overlong_user_messages() {
        // Use a small truncation limit so the test remains fast while still validating
        // that oversized user content is truncated.
        let max_tokens = 16;
        let big = "word ".repeat(200);
        let history = super::build_compacted_history_with_limit(
            Vec::new(),
            std::slice::from_ref(&big),
            "SUMMARY",
            max_tokens,
        );
        assert_eq!(history.len(), 2);

        let truncated_message = &history[0];
        let summary_message = &history[1];

        let truncated_text = match truncated_message {
            ResponseItem::Message { role, content, .. } if role == "user" => {
                content_items_to_text(content).unwrap_or_default()
            }
            other => panic!("unexpected item in history: {other:?}"),
        };

        assert!(
            truncated_text.contains("tokens truncated"),
            "expected truncation marker in truncated user message"
        );
        assert!(
            !truncated_text.contains(&big),
            "truncated user message should not include the full oversized user text"
        );

        let summary_text = match summary_message {
            ResponseItem::Message { role, content, .. } if role == "user" => {
                content_items_to_text(content).unwrap_or_default()
            }
            other => panic!("unexpected item in history: {other:?}"),
        };
        assert_eq!(summary_text, "SUMMARY");
    }

    #[test]
    fn build_token_limited_compacted_history_appends_summary_message() {
        let initial_context: Vec<ResponseItem> = Vec::new();
        let user_messages = vec!["first user message".to_string()];
        let summary_text = "summary text";

        let history = build_compacted_history(initial_context, &user_messages, summary_text);
        assert!(
            !history.is_empty(),
            "expected compacted history to include summary"
        );

        let last = history.last().expect("history should have a summary entry");
        let summary = match last {
            ResponseItem::Message { role, content, .. } if role == "user" => {
                content_items_to_text(content).unwrap_or_default()
            }
            other => panic!("expected summary message, found {other:?}"),
        };
        assert_eq!(summary, summary_text);
    }

    /// Reproduce the parallel tool call boundary bug: with items
    /// [Msg, Call_A, Call_B, Output_A, Output_B], the split must not
    /// orphan Call_A in old_half while its Output_A lands in recent_half.
    #[test]
    fn split_does_not_orphan_parallel_tool_calls() {
        use codex_protocol::models::FunctionCallOutputBody;
        use codex_protocol::models::FunctionCallOutputPayload;

        let user_msg = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "x".repeat(4000), // ~1000 tokens
            }],
            end_turn: None,
            phase: None,
        };
        let call_a = ResponseItem::FunctionCall {
            id: None,
            name: "tool_a".to_string(),
            arguments: "{}".to_string(),
            call_id: "call_a".to_string(),
        };
        let call_b = ResponseItem::FunctionCall {
            id: None,
            name: "tool_b".to_string(),
            arguments: "{}".to_string(),
            call_id: "call_b".to_string(),
        };
        let output_a = ResponseItem::FunctionCallOutput {
            call_id: "call_a".to_string(),
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text("result_a".to_string()),
                success: Some(true),
            },
        };
        let output_b = ResponseItem::FunctionCallOutput {
            call_id: "call_b".to_string(),
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text("result_b".to_string()),
                success: Some(true),
            },
        };

        let items = vec![user_msg, call_a, call_b, output_a, output_b];

        // Compute per-item token counts and pick a fraction that forces
        // the split right after Call_B (index 2), i.e. split_idx = 3.
        let token_counts: Vec<i64> = items.iter().map(|i| estimate_item_token_count(i)).collect();
        let total: i64 = token_counts.iter().sum();
        let after_call_b: i64 = token_counts[..3].iter().sum();

        // fraction such that target = after_call_b, so accumulated hits
        // target exactly when Call_B is processed.
        let fraction = after_call_b as f64 / total as f64;

        let result = split_history_at_token_midpoint(&items, fraction);
        let (old, recent) = result.expect("should produce a split");

        // Every call in old must have its output in old too.
        for item in &old {
            if let ResponseItem::FunctionCall { call_id, .. } = item {
                assert!(
                    old.iter().any(|o| matches!(
                        o,
                        ResponseItem::FunctionCallOutput { call_id: cid, .. } if cid == call_id
                    )),
                    "Call {call_id} is in old half but its output is missing from old half"
                );
            }
        }
        // Every output in recent must have its call in recent too.
        for item in &recent {
            if let ResponseItem::FunctionCallOutput { call_id, .. } = item {
                assert!(
                    recent.iter().any(|c| matches!(
                        c,
                        ResponseItem::FunctionCall { call_id: cid, .. } if cid == call_id
                    )),
                    "Output for {call_id} is in recent half but its call is missing from recent half"
                );
            }
        }
    }

    #[test]
    fn split_ignores_reinjected_scaffolding_when_finding_midpoint() {
        let user_instructions = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: format!(
                    "# AGENTS.md instructions for /tmp\n\n<INSTRUCTIONS>\n{}\n</INSTRUCTIONS>",
                    "x".repeat(14_000)
                ),
            }],
            end_turn: None,
            phase: None,
        };
        let env_context = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "<environment_context>\n<cwd>/tmp</cwd>\n<shell>zsh</shell>\n</environment_context>"
                    .to_string(),
            }],
            end_turn: None,
            phase: None,
        };
        let older_user = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "u".repeat(8_000),
            }],
            end_turn: None,
            phase: None,
        };
        let older_assistant = ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "a".repeat(8_000),
            }],
            end_turn: None,
            phase: None,
        };
        let recent_user = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "r".repeat(8_000),
            }],
            end_turn: None,
            phase: None,
        };
        let items = vec![
            user_instructions,
            env_context,
            older_user.clone(),
            older_assistant,
            recent_user.clone(),
        ];

        let (old_half, recent_half) = split_history_at_token_midpoint(&items, 0.5)
            .expect("expected a split over compactable conversation items");

        assert!(
            old_half.contains(&older_user),
            "midpoint should advance into conversational history, not stop inside reinjected scaffolding"
        );
        assert!(
            recent_half.contains(&recent_user),
            "recent conversational items should remain in the recent half"
        );
    }

    #[test]
    fn split_returns_none_when_only_reinjected_scaffolding_exists() {
        let items = vec![
            ResponseItem::Message {
                id: None,
                role: "developer".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "policy".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "# AGENTS.md instructions for /tmp\n\n<INSTRUCTIONS>\nDo X\n</INSTRUCTIONS>"
                        .to_string(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "<environment_context>\n<cwd>/tmp</cwd>\n<shell>zsh</shell>\n</environment_context>"
                        .to_string(),
                }],
                end_turn: None,
                phase: None,
            },
        ];

        assert!(
            split_history_at_token_midpoint(&items, 0.5).is_none(),
            "scaffolding-only history should skip partial split and fall back to full compaction"
        );
    }

    #[test]
    fn build_compacted_history_preserves_turn_aborted_markers() {
        let marker = format!(
            "{TURN_ABORTED_OPEN_TAG}\n  <turn_id>turn-1</turn_id>\n  <reason>interrupted</reason>\n</turn_aborted>"
        );
        let items = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: marker.clone(),
                }],
                end_turn: None,
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "real user message".to_string(),
                }],
                end_turn: None,
                phase: None,
            },
        ];

        let user_messages = collect_user_messages(&items);
        let history = build_compacted_history(Vec::new(), &user_messages, "SUMMARY");

        let found_marker = history.iter().any(|item| match item {
            ResponseItem::Message { role, content, .. } if role == "user" => {
                content_items_to_text(content).is_some_and(|text| text == marker)
            }
            _ => false,
        });
        assert!(
            found_marker,
            "expected compacted history to retain <turn_aborted> marker"
        );
    }
}
