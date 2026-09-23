# Static Assets

Vendored JavaScript-Lib for client based markdown rendering of chat messages.
The files are taken over unchanged from their official release.
They are delivered via the mount `/static` by `webapp.py`.

| File             | Lib.       | Version | License             | Source |
|------------------|------------|---------|---------------------|--------|
| `marked.min.js`  | marked     | 15.0.12 | MIT                 | https://cdn.jsdelivr.net/npm/marked@15.0.12/marked.min.js |
| `purify.min.js`  | DOMPurify  | 3.2.6   | Apache-2.0 / MPL-2.0 | https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js |

For an update both files must be replaced and the Tests in `tests/test_webapp.py` be updated.
