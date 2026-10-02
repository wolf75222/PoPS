# Independent Source review of 295cc73

Exact clean author freeze 295cc73bc0e7457057ef0d0c1905fea1ef4d5f11. 14 PASS0.79s: author13 plus independent peer-only rejection branch. No Native/MPI execution.

Actual capture no longer invokes integrity admission. All ranks first write their actual local bytes under collective_call; ROOT writes complete bytes/valid NPY/clock/storage metadata after that successful boundary. Partial receipt is persisted next. Only then guard_persisted_storage votes validation before returning to the next run. Local original errors are rethrown; a peer-only failure journals the same gathered failures and throws, preventing future operations. No payload is replaced, omitted, normalized or synthesized.

I/O or ledger failure remains fail-closed and can supersede an integrity error; this review does not promise successful persistence on filesystem failure. The test transport adversary is explicitly Source-only, not evidence of real MPI deadlock freedom. Physics, thresholds and historical fixture@2 are unchanged. ROOT must execute the new profile on an authentic SDK20.
