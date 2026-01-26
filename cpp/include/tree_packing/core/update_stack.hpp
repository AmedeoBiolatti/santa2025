#pragma once

#include "tree_packing/core/types.hpp"
#include <vector>
#include <cstdint>

namespace tree_packing {

// Forward declarations
class Problem;
struct SolutionEval;

// Type of update operation
enum class UpdateType : uint8_t {
    Insert,  // Tree was inserted (was invalid, now valid)
    Update,  // Tree params changed (was valid, still valid)
    Remove   // Tree was removed (was valid, now invalid)
};

// Single update entry - stores enough info to undo the operation
struct UpdateEntry {
    int index;              // Tree index that was modified
    UpdateType type;        // Type of operation
    TreeParams prev_params; // Previous params (for Update/Remove, to restore)

    UpdateEntry() = default;
    UpdateEntry(int idx, UpdateType t, TreeParams prev = {})
        : index(idx), type(t), prev_params(prev) {}
};

// Stack of updates with pre-allocated capacity
// Supports marking checkpoints and reverting to them
class UpdateStack {
public:
    explicit UpdateStack(size_t capacity = 64)
        : entries_(capacity)
        , size_(0)
    {}

    // Reset stack (no deallocation)
    void clear() { size_ = 0; }

    // Current number of entries
    [[nodiscard]] size_t size() const { return size_; }

    // Check if empty
    [[nodiscard]] bool empty() const { return size_ == 0; }

    // Capacity
    [[nodiscard]] size_t capacity() const { return entries_.size(); }

    // Push an insert operation (tree at index was inserted)
    void push_insert(int index) {
        ensure_capacity();
        entries_[size_++] = UpdateEntry(index, UpdateType::Insert);
    }

    // Push an update operation (tree at index was updated from prev_params)
    void push_update(int index, TreeParams prev_params) {
        ensure_capacity();
        entries_[size_++] = UpdateEntry(index, UpdateType::Update, prev_params);
    }

    // Push a remove operation (tree at index was removed, had prev_params)
    void push_remove(int index, TreeParams prev_params) {
        ensure_capacity();
        entries_[size_++] = UpdateEntry(index, UpdateType::Remove, prev_params);
    }

    // Get entry at position (0 = oldest, size-1 = newest)
    [[nodiscard]] const UpdateEntry& operator[](size_t i) const {
        return entries_[i];
    }

    // Get the top entry
    [[nodiscard]] const UpdateEntry& top() const {
        return entries_[size_ - 1];
    }

    // Pop n entries (just decrements size, no deallocation)
    void pop(size_t n = 1) {
        if (n >= size_) {
            size_ = 0;
        } else {
            size_ -= n;
        }
    }

    // Mark current position (returns current size as checkpoint)
    [[nodiscard]] size_t mark() const { return size_; }

    // Get number of entries since a checkpoint
    [[nodiscard]] size_t entries_since(size_t checkpoint) const {
        return (size_ > checkpoint) ? (size_ - checkpoint) : 0;
    }

    // Iterate over entries in reverse (newest to oldest) from current position to checkpoint
    // Callback signature: void(const UpdateEntry&)
    template<typename Callback>
    void for_each_reverse(size_t checkpoint, Callback&& cb) const {
        for (size_t i = size_; i > checkpoint; --i) {
            cb(entries_[i - 1]);
        }
    }

    // Iterate over entries forward (oldest to newest) from checkpoint to current
    template<typename Callback>
    void for_each_forward(size_t checkpoint, Callback&& cb) const {
        for (size_t i = checkpoint; i < size_; ++i) {
            cb(entries_[i]);
        }
    }

    // Reserve more capacity if needed
    void reserve(size_t new_capacity) {
        if (new_capacity > entries_.size()) {
            entries_.resize(new_capacity);
        }
    }

private:
    void ensure_capacity() {
        if (size_ >= entries_.size()) {
            entries_.resize(entries_.size() * 2);
        }
    }

    std::vector<UpdateEntry> entries_;
    size_t size_;
};

// Helper to apply rollback using the update stack
// Reverts all entries from current position back to checkpoint
// Returns the new intersection violation (or you can recompute)
void apply_rollback(
    const Problem& problem,
    SolutionEval& solution,
    UpdateStack& stack,
    size_t checkpoint
);

// Helper to collect indices and params for batch operations during rollback
struct RollbackBatch {
    std::vector<int> insert_indices;
    TreeParamsSoA insert_params;

    std::vector<int> update_indices;
    TreeParamsSoA update_params;

    std::vector<int> remove_indices;

    void clear() {
        insert_indices.clear();
        insert_params.clear();
        update_indices.clear();
        update_params.clear();
        remove_indices.clear();
    }

    void reserve(size_t n) {
        insert_indices.reserve(n);
        insert_params.reserve(n);
        update_indices.reserve(n);
        update_params.reserve(n);
        remove_indices.reserve(n);
    }
};

// Build a rollback batch from stack entries (checkpoint to current)
// Groups operations by type for efficient batch application
void build_rollback_batch(
    const UpdateStack& stack,
    size_t checkpoint,
    RollbackBatch& batch
);

}  // namespace tree_packing