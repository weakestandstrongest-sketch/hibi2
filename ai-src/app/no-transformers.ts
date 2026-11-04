// まいにち uses only the WebLLM backend. The Transformers.js backend's library is replaced by this stub,
// so its 600 KB chunk and 28 MB runtime are not shipped (the engine imports it only when that backend is used).
throw new Error("このアプリでは Transformers.js のモデルは使えません");
export {};
