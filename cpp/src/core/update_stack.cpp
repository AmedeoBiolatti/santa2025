#include "tree_packing/core/update_stack.hpp"
#include "tree_packing/core/problem.hpp"
#include "tree_packing/core/solution.hpp"

namespace tree_packing {

void build_rollback_batch(
    const UpdateStack& stack,
    size_t checkpoint,
    RollbackBatch& batch
) {
    batch.clear();
    size_t n = stack.entries_since(checkpoint);
    batch.reserve(n);

    // Process in reverse order (newest to oldest) to correctly undo
    stack.for_each_reverse(checkpoint, [&](const UpdateEntry& entry) {
        switch (entry.type) {
            case UpdateType::Insert:
                // To undo an insert, we need to remove
                batch.remove_indices.push_back(entry.index);
                break;

            case UpdateType::Update:
                // To undo an update, we need to update back to prev_params
                batch.update_indices.push_back(entry.index);
                batch.update_params.push_back(entry.prev_params);
                break;

            case UpdateType::Remove:
                // To undo a remove, we need to insert with prev_params
                batch.insert_indices.push_back(entry.index);
                batch.insert_params.push_back(entry.prev_params);
                break;
        }
    });
}

void apply_rollback(
    const Problem& problem,
    SolutionEval& solution,
    UpdateStack& stack,
    size_t checkpoint
) {
    if (stack.size() <= checkpoint) {
        return;  // Nothing to rollback
    }

    // IMPORTANT: We must apply operations in strict reverse order!
    // Batching by type is incorrect when the same index appears in multiple operations.
    //
    // Example: Remove(3), Insert(3), Update(3)
    // If batched: remove 3, update 3 (BUG: invalid!), insert 3
    // If strict reverse: undo Update, undo Insert, undo Remove - correct!

    // Scratch buffers for single operations
    thread_local std::vector<int> single_index;
    thread_local TreeParamsSoA single_params;

    single_index.resize(1);
    single_params.resize(1);

    size_t n = stack.entries_since(checkpoint);

    // Process in reverse order (newest to oldest) to restore params
    // We use the Problem API which does incremental updates
    stack.for_each_reverse(checkpoint, [&](const UpdateEntry& entry) {
        single_index[0] = entry.index;

        switch (entry.type) {
            case UpdateType::Insert:
                // To undo an insert, remove the tree
                problem.remove_and_eval(solution, single_index);
                break;

            case UpdateType::Update:
                // To undo an update, restore the old params
                single_params.set(0, entry.prev_params);
                problem.update_and_eval(solution, single_index, single_params);
                break;

            case UpdateType::Remove:
                // To undo a remove, insert with the old params
                single_params.set(0, entry.prev_params);
                problem.insert_and_eval(solution, single_index, single_params);
                break;
        }
    });

    // Pop the entries we just rolled back
    stack.pop(n);
}

}  // namespace tree_packing