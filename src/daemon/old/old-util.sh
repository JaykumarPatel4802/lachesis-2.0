#!/bin/bash

# Output file path
OUTPUT_FILE=stats.log

# Stream docker stats to the output file continuously
while true; do
  # Get the current timestamp in ISO 8601 format with milliseconds and UTC timezone indicator
  TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%S.%3NZ')

  # Write the timestamp and docker stats to the output file
  echo "$TIMESTAMP" >> "$OUTPUT_FILE"
  docker stats --no-stream >> "$OUTPUT_FILE"
  echo " " >> "$OUTPUT_FILE"
done
