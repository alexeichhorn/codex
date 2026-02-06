use std::sync::Arc;

use crate::Prompt;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::compact::COMPACT_SPLIT_FRACTION;
use crate::compact::split_history_at_token_midpoint;
use crate::context_manager::ContextManager;
use crate::context_manager::is_codex_generated_item;
use crate::error::Result as CodexResult;
use crate::protocol::CompactedItem;
use crate::protocol::EventMsg;
use crate::protocol::RolloutItem;
use crate::protocol::TurnStartedEvent;
use codex_protocol::items::ContextCompactionItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;
use tracing::info;

pub(crate) async fn run_inline_remote_auto_compact_task(
    sess: Arc<Session>,
    turn_context: Arc<TurnContext>,
) {
    run_partial_remote_compact_task_inner(&sess, &turn_context).await;
}

pub(crate) async fn run_remote_compact_task(sess: Arc<Session>, turn_context: Arc<TurnContext>) {
    let start_event = EventMsg::TurnStarted(TurnStartedEvent {
        model_context_window: turn_context.model_context_window(),
        collaboration_mode_kind: turn_context.collaboration_mode.mode,
    });
    sess.send_event(&turn_context, start_event).await;

    run_remote_compact_task_inner(&sess, &turn_context).await;
}

async fn run_remote_compact_task_inner(sess: &Arc<Session>, turn_context: &Arc<TurnContext>) {
    if let Err(err) = run_remote_compact_task_inner_impl(sess, turn_context).await {
        let event = EventMsg::Error(
            err.to_error_event(Some("Error running remote compact task".to_string())),
        );
        sess.send_event(turn_context, event).await;
    }
}

/// Partial remote compaction: only send the older half of history to the
/// compact endpoint, keeping the recent half verbatim.
async fn run_partial_remote_compact_task_inner(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
) {
    if let Err(err) =
        run_partial_remote_compact_task_inner_impl(sess, turn_context).await
    {
        let event = EventMsg::Error(
            err.to_error_event(Some("Error running remote compact task".to_string())),
        );
        sess.send_event(turn_context, event).await;
    }
}

async fn run_partial_remote_compact_task_inner_impl(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
) -> CodexResult<()> {
    let full_history = sess.clone_history().await;
    let all_items = full_history.raw_items().to_vec();

    // Try to split; if old half is too small, fall back to full remote compaction.
    let Some((old_half, recent_half)) =
        split_history_at_token_midpoint(&all_items, COMPACT_SPLIT_FRACTION)
    else {
        return run_remote_compact_task_inner_impl(sess, turn_context).await;
    };

    let compaction_item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(turn_context, &compaction_item)
        .await;

    let base_instructions = sess.get_base_instructions().await;

    // Ghost snapshots from the old half need preservation.
    let ghost_snapshots: Vec<ResponseItem> = old_half
        .iter()
        .filter(|item| matches!(item, ResponseItem::GhostSnapshot { .. }))
        .cloned()
        .collect();

    // Build a ContextManager from just the old half, trimming if needed.
    let mut old_half_cm = ContextManager::new();
    old_half_cm.replace(old_half);
    let deleted_items = trim_function_call_history_to_fit_context_window(
        &mut old_half_cm,
        turn_context.as_ref(),
        &base_instructions,
    );
    if deleted_items > 0 {
        info!(
            turn_id = %turn_context.sub_id,
            deleted_items,
            "trimmed history items before partial remote compaction"
        );
    }

    let prompt = Prompt {
        input: old_half_cm.for_prompt(),
        tools: vec![],
        parallel_tool_calls: false,
        base_instructions,
        personality: turn_context.personality,
        output_schema: None,
    };

    let mut compacted_old = sess
        .services
        .model_client
        .compact_conversation_history(
            &prompt,
            &turn_context.model_info,
            &turn_context.otel_manager,
        )
        .await?;

    // Stitch together: compacted old half + recent half verbatim + ghost snapshots.
    compacted_old.extend(recent_half);
    if !ghost_snapshots.is_empty() {
        compacted_old.extend(ghost_snapshots);
    }

    sess.replace_history(compacted_old.clone()).await;
    sess.recompute_token_usage(turn_context).await;

    let compacted_item = CompactedItem {
        message: String::new(),
        replacement_history: Some(compacted_old),
    };
    sess.persist_rollout_items(&[RolloutItem::Compacted(compacted_item)])
        .await;

    sess.emit_turn_item_completed(turn_context, compaction_item)
        .await;
    Ok(())
}

async fn run_remote_compact_task_inner_impl(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
) -> CodexResult<()> {
    let compaction_item = TurnItem::ContextCompaction(ContextCompactionItem::new());
    sess.emit_turn_item_started(turn_context, &compaction_item)
        .await;
    let mut history = sess.clone_history().await;
    let base_instructions = sess.get_base_instructions().await;
    let deleted_items = trim_function_call_history_to_fit_context_window(
        &mut history,
        turn_context.as_ref(),
        &base_instructions,
    );
    if deleted_items > 0 {
        info!(
            turn_id = %turn_context.sub_id,
            deleted_items,
            "trimmed history items before remote compaction"
        );
    }

    // Required to keep `/undo` available after compaction
    let ghost_snapshots: Vec<ResponseItem> = history
        .raw_items()
        .iter()
        .filter(|item| matches!(item, ResponseItem::GhostSnapshot { .. }))
        .cloned()
        .collect();

    let prompt = Prompt {
        input: history.for_prompt(),
        tools: vec![],
        parallel_tool_calls: false,
        base_instructions,
        personality: turn_context.personality,
        output_schema: None,
    };

    let mut new_history = sess
        .services
        .model_client
        .compact_conversation_history(
            &prompt,
            &turn_context.model_info,
            &turn_context.otel_manager,
        )
        .await?;

    if !ghost_snapshots.is_empty() {
        new_history.extend(ghost_snapshots);
    }
    sess.replace_history(new_history.clone()).await;
    sess.recompute_token_usage(turn_context).await;

    let compacted_item = CompactedItem {
        message: String::new(),
        replacement_history: Some(new_history),
    };
    sess.persist_rollout_items(&[RolloutItem::Compacted(compacted_item)])
        .await;

    sess.emit_turn_item_completed(turn_context, compaction_item)
        .await;
    Ok(())
}

fn trim_function_call_history_to_fit_context_window(
    history: &mut ContextManager,
    turn_context: &TurnContext,
    base_instructions: &BaseInstructions,
) -> usize {
    let mut deleted_items = 0usize;
    let Some(context_window) = turn_context.model_context_window() else {
        return deleted_items;
    };

    while history
        .estimate_token_count_with_base_instructions(base_instructions)
        .is_some_and(|estimated_tokens| estimated_tokens > context_window)
    {
        let Some(last_item) = history.raw_items().last() else {
            break;
        };
        if !is_codex_generated_item(last_item) {
            break;
        }
        if !history.remove_last_item() {
            break;
        }
        deleted_items += 1;
    }

    deleted_items
}
