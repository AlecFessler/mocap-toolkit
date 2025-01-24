const cache_line_size = 64;

pub fn Queue(comptime T: type) type {
    return extern struct {
        const Self = @This();

        head: usize,
        cached_tail: usize,
        capacity1: usize,
        buffer1: [*]T,
        padding1: [cache_line_size]u8 align(cache_line_size),

        tail: usize,
        cached_head: usize,
        capacity2: usize,
        buffer2: [*]T,
        padding2: [cache_line_size]u8 align(cache_line_size),

        pub fn init(self: *Self, buffer: []T) void {
            self.head = 0;
            self.cached_tail = 0;
            self.capacity1 = buffer.len;
            self.buffer1 = buffer.ptr;
            self.padding1 = undefined;

            self.tail = 0;
            self.cached_head = 0;
            self.capacity2 = buffer.len;
            self.buffer2 = buffer.ptr;
            self.padding2 = undefined;
        }

        /// Enqueue operation strictly reads from the top half of the struct
        /// except for checking if the queue is full
        pub fn enqueue(self: *Self, data: T) bool {
            // can safely read head without atomic load
            var next = self.head + 1;
            if (next == self.capacity1) next = 0;

            if (next == self.cached_tail) { // check if queue appears full
                self.cached_tail = @atomicLoad(usize, &self.tail, .acquire);
                if (next == self.cached_tail) return false; // queue full
            }

            self.buffer1[self.head] = data;
            @atomicStore(usize, &self.head, next, .release);

            return true;
        }

        // Dequeue operation strictly reads from the bottom half of the struct
        // except for checking if the queue is empty
        pub fn dequeue(self: *Self) ?T {
            // can safely read tail without atomic load
            if (self.tail == self.cached_head) { // check if the queue appears empty
                self.cached_head = @atomicLoad(usize, &self.head, .acquire);
                if (self.tail == self.cached_head) return null; // queue empty
            }

            const data = self.buffer2[self.tail];

            var next = self.tail + 1;
            if (next == self.capacity2) next = 0;

            @atomicStore(usize, &self.tail, next, .release);

            return data;
        }
    };
}
