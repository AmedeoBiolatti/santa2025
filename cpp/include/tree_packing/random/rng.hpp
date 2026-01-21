#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>

namespace tree_packing {

// PCG64 random number generator (permuted congruential generator)
// Fast, high-quality, and splittable
class RNG {
public:
    RNG() : state_(0x853c49e6748fea9bULL), inc_(0xda3e39cb94b95bdbULL) {}
    explicit RNG(uint64_t seed) : state_(0), inc_(seed | 1) {
        (void)next();  // Discard for seeding
        state_ += seed;
        (void)next();  // Discard for seeding
    }

    // Generate next random number
    [[nodiscard]] uint64_t next() {
        uint64_t oldstate = state_;
        state_ = oldstate * 6364136223846793005ULL + inc_;
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    // Generate random float in [0, 1)
    [[nodiscard]] float uniform() {
        return static_cast<float>(next()) / static_cast<float>(1ULL << 32);
    }

    // Generate random float in [min, max)
    [[nodiscard]] float uniform(float min, float max) {
        return min + uniform() * (max - min);
    }

    // Generate random integer in [min, max] (inclusive)
    [[nodiscard]] int randint(int min, int max) {
        if (min > max) std::swap(min, max);
        uint64_t range = static_cast<uint64_t>(max - min) + 1;
        return min + static_cast<int>(next() % range);
    }

    // Generate random permutation of [0, n)
    [[nodiscard]] std::vector<int> permutation(int n) {
        std::vector<int> result(n);
        for (int i = 0; i < n; ++i) result[i] = i;
        for (int i = n - 1; i > 0; --i) {
            int j = randint(0, i);
            std::swap(result[i], result[j]);
        }
        return result;
    }

    // Choose k random indices from [0, n) without replacement
    [[nodiscard]] std::vector<int> choice(int n, int k) {
        if (k > n) k = n;
        auto perm = permutation(n);
        perm.resize(k);
        return perm;
    }

    // Choose index based on weights (unnormalized probabilities)
    [[nodiscard]] int weighted_choice(const std::vector<float>& weights) {
        float total = 0.0f;
        for (float w : weights) total += w;

        float r = uniform() * total;
        float cumsum = 0.0f;
        for (size_t i = 0; i < weights.size(); ++i) {
            cumsum += weights[i];
            if (r < cumsum) return static_cast<int>(i);
        }
        return static_cast<int>(weights.size() - 1);
    }

    // Split the RNG (return a new independent RNG seeded from current state)
    [[nodiscard]] RNG split() {
        return RNG(next());
    }

    // Get current state for serialization
    [[nodiscard]] uint64_t state() const { return state_; }
    [[nodiscard]] uint64_t increment() const { return inc_; }

    // Set state for deserialization
    void set_state(uint64_t state, uint64_t inc) {
        state_ = state;
        inc_ = inc;
    }

private:
    uint64_t state_;
    uint64_t inc_;
};

}  // namespace tree_packing
