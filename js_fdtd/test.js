import fs from 'fs';
process.on('uncaughtException', err => {
    fs.writeFileSync('error_trace.txt', err.stack || err.message);
    process.exit(1);
});
process.on('unhandledRejection', err => {
    fs.writeFileSync('error_trace.txt', err.stack || err.message);
    process.exit(1);
});
import('./src/index.js').catch(err => {
    fs.writeFileSync('error_trace.txt', err.stack || err.message);
});
