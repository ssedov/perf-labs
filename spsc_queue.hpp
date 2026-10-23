#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include "constants.h"

namespace spsc {

// One thread calls try_push(); one other thread consumes messages.
// The callback's byte span remains valid only for the duration of the callback.
template <std::size_t Capacity, std::size_t SlotSize = 64>
class Queue2 {
  static_assert(Capacity > 0);
  static_assert(SlotSize > 0);
  static_assert(Capacity <= std::numeric_limits<std::size_t>::max() / SlotSize);

 public:
  static constexpr std::size_t capacity = Capacity;
  static constexpr std::size_t slot_size = SlotSize;
  static constexpr std::size_t max_message_bytes = Capacity * SlotSize;

  Queue2() noexcept = default;
  Queue2(const Queue2&) = delete;
  Queue2& operator=(const Queue2&) = delete;

  bool try_push(std::span<const std::byte> message) noexcept {
    if (message.size() > max_message_bytes) {
      return false;
    }

    const auto write = write_pos_.load(std::memory_order_relaxed);
    const auto read = read_pos_.load(std::memory_order_acquire);
    const auto required = slots_required(message.size());
    const auto index = write % Capacity;
    const auto remaining = Capacity - index;
    const auto skipped =
        required < Capacity && required > remaining ? remaining : 0;
    if (required > Capacity - skipped ||
        required + skipped > Capacity - (write - read)) {
      return false;
    }

    auto& header = headers_[index];
    header.size = message.size();
    header.data = skipped || required == Capacity ? buffer_[0].data
                                                   : buffer_[index].data;
    header.slots = required + skipped;
    if (!message.empty()) {
      std::memcpy(header.data, message.data(), message.size());
    }
    write_pos_.store(write + header.slots, std::memory_order_release);
    return true;
  }

  template <class Consumer>
  bool consume(Consumer&& consumer) noexcept {
    static_assert(std::is_nothrow_invocable_v<Consumer, std::span<const std::byte>>,
                  "consumer callback must be noexcept");

    const auto read = read_pos_.load(std::memory_order_relaxed);
    if (read == write_pos_.load(std::memory_order_acquire)) {
      return false;
    }
    auto& header = headers_[read % Capacity];
    std::forward<Consumer>(consumer)(
        std::span<const std::byte>{header.data, header.size});
    read_pos_.store(read + header.slots, std::memory_order_release);
    return true;
  }

  template <class Consumer>
  bool try_consume(Consumer&& consumer) noexcept {
    return consume(std::forward<Consumer>(consumer));
  }

  std::span<const std::byte> peek() noexcept {
    const auto read = read_pos_.load(std::memory_order_relaxed);
    if (read == write_pos_.load(std::memory_order_acquire)) {
      return {};
    }
    auto& header = headers_[read % Capacity];
    return {header.data, header.size};
  }

  std::span<const std::byte> front() noexcept { return peek(); }

  bool commit() noexcept {
    const auto read = read_pos_.load(std::memory_order_relaxed);
    if (read == write_pos_.load(std::memory_order_acquire)) {
      return false;
    }
    read_pos_.store(read + headers_[read % Capacity].slots,
                    std::memory_order_release);
    return true;
  }

 private:
  struct Slot {
    std::byte data[SlotSize]{};
  };

  struct alignas(cache_line) Message {
    std::size_t size{0};
    std::byte* data{nullptr};
    std::size_t slots{0};
  };

  Slot buffer_[Capacity]{};
  Message headers_[Capacity]{};
  alignas(cache_line) std::atomic<std::size_t> write_pos_{0};
  alignas(cache_line) std::atomic<std::size_t> read_pos_{0};

  static constexpr std::size_t slots_required(std::size_t bytes) noexcept {
    return bytes == 0 ? 1 : 1 + (bytes - 1) / SlotSize;
  }
};

}  // namespace spsc
