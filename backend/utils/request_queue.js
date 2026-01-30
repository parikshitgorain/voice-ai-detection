class QueueFullError extends Error {
  constructor(queueLength, maxQueue) {
    super("Queue is full.");
    this.code = "QUEUE_FULL";
    this.queueLength = queueLength;
    this.maxQueue = maxQueue;
  }
}

const createRequestQueue = (options = {}) => {
  const maxConcurrent = Number.isFinite(options.maxConcurrent)
    ? options.maxConcurrent
    : 3;
  const maxQueue = Number.isFinite(options.maxQueue) ? options.maxQueue : 10;
  let active = 0;
  const queue = [];

  const getStats = () => ({
    active,
    queued: queue.length,
    maxConcurrent,
    maxQueue,
  });

  const removeQueued = (id) => {
    const idx = queue.findIndex((item) => item.id === id);
    if (idx >= 0) {
      queue.splice(idx, 1);
      return true;
    }
    return false;
  };

  const startItem = (item) => {
    active += 1;
    item.startedAt = Date.now();
    if (item.req) {
      item.req.queueMeta = {
        queued: item.queued,
        position: item.position,
        queuedAt: item.queuedAt,
        startedAt: item.startedAt,
      };
    }

    const done = () => {
      active = Math.max(0, active - 1);
      if (queue.length > 0) {
        const next = queue.shift();
        startItem(next);
      }
    };

    Promise.resolve()
      .then(item.task)
      .catch(item.onError)
      .finally(done);
  };

  const enqueue = ({ id, task, req, onError }) => {
    const safeOnError = typeof onError === "function" ? onError : () => {};
    const queuedAt = Date.now();
    const item = {
      id,
      task,
      req,
      onError: safeOnError,
      queued: false,
      queuedAt,
      position: 0,
      startedAt: null,
    };

    if (active < maxConcurrent) {
      startItem(item);
      return { queued: false, position: 0, queuedAt };
    }

    if (queue.length >= maxQueue) {
      throw new QueueFullError(queue.length, maxQueue);
    }

    item.queued = true;
    item.position = queue.length + 1;
    queue.push(item);

    if (req) {
      req.queueMeta = {
        queued: true,
        position: item.position,
        queuedAt: item.queuedAt,
        startedAt: null,
      };
      req.on("aborted", () => {
        removeQueued(id);
      });
    }

    return { queued: true, position: item.position, queuedAt };
  };

  return { enqueue, getStats };
};

module.exports = {
  QueueFullError,
  createRequestQueue,
};
