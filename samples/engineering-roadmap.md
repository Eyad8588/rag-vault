# Engineering Roadmap Q1 2027 (internal)

For internal use only. This roadmap is shared with the Orion Labs engineering and product teams.

## Platform team
The platform team, led by Omar Khaled, will migrate the Nimbus sync service from PostgreSQL logical replication to an event-sourced design built on Apache Kafka. The goal is to cut sync latency below 200 ms at the 95th percentile.

## Mobile team
Mona Hassan's mobile team ships the offline editor for Android first, then iOS. The team will adopt Kotlin Multiplatform for the shared sync engine.

## Search
The data team is evaluating LanceDB for semantic search inside Nimbus notes. A prototype uses the qwen3-embedding model running on-device.

## Risks
Hiring two senior backend engineers in Berlin is behind schedule. The Kafka migration depends on the new on-call rotation being in place.
