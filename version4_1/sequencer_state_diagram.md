# Sequencer State Diagram

```mermaid
stateDiagram-v2
    [*] --> S_IDLE

    S_IDLE --> S_WGT_LEAD: i_size_ld
    S_IDLE --> S_IDLE: !i_size_ld

    S_WGT_LEAD --> S_WGT_LEAD: !wgt_stream_en
    S_WGT_LEAD --> S_STREAM: wgt_stream_en && i_act_stream_valid
    S_WGT_LEAD --> S_REPLAY_WAIT: wgt_stream_en && !i_act_stream_valid

    S_STREAM --> S_STREAM: normal stream
    S_STREAM --> S_REPLAY_WAIT: wgt_done && more wgt groups && !act_done
    S_STREAM --> S_DRAIN: act_done

    S_REPLAY_WAIT --> S_REPLAY_WAIT: !(i_act_stream_valid && i_wgt_stream_valid)
    S_REPLAY_WAIT --> S_STREAM: i_act_stream_valid && i_wgt_stream_valid

    S_DRAIN --> S_DRAIN: act_out_en || wgt_out_en
    S_DRAIN --> S_WGT_LEAD: drained && more act groups
    S_DRAIN --> S_DONE: drained && last act group

    S_DONE --> S_IDLE
```

## State Notes

| State | Meaning |
| --- | --- |
| `S_IDLE` | Wait for `i_size_ld`; latch matrix sizes and base addresses. |
| `S_WGT_LEAD` | Let weight stream enter one beat earlier for systolic-array skew. |
| `S_STREAM` | Normal act/wgt synchronized stream; pipeline advances only on valid handshakes. |
| `S_REPLAY_WAIT` | Hold the pipeline while waiting for both act and replayed wgt stream to be valid. |
| `S_DRAIN` | Stop new SRAM streams and flush remaining skewed act/wgt data through the array. |
| `S_DONE` | Transaction complete. |
