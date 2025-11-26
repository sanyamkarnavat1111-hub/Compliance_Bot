# RFP Completeness System - Startup Guide

## Overview

The RFP Completeness System can be run in two different modes:

1. **Service Mode** - Runs continuously with workers polling the database for jobs
2. **Single Job Mode** - Processes a single RFP completeness check and exits

## Service Mode (Database Polling)

This mode is used when you have a database with jobs that need to be processed continuously.

### Starting the Service

```bash
python manager.py
```

This will:
- Start the manager
- Initialize worker threads (as configured in `config.py`)
- Workers will poll the database for jobs
- Process jobs as they become available
- Workers will automatically shut down after 5 minutes of inactivity (configurable)

### Configuration

Edit `config.py` to adjust:
- `COMPLETENESS_WORKERS`: Number of completeness check workers
- `PROPOSAL_WORKERS`: Number of proposal evaluation workers
- `WORKER_IDLE_TIMEOUT_SECONDS`: Idle timeout (0 to disable)
- `WORKER_POLL_INTERVAL_SECONDS`: How often to check for jobs

## Single Job Mode (Direct Execution)

Use this mode to process a single RFP completeness check without database polling.

### Option 1: Using the dedicated script

```bash
python run_single_job.py \
    --rfp-url "https://example.com/rfp.pdf" \
    --ea-standard-url "https://example.com/ea-standard.pdf" \
    --model openai \
    --output-language english
```

### Option 2: Direct Python execution

```python
from main import RFPCompleteness

rfp_completeness = RFPCompleteness()
id, result, error_user, error_tech = rfp_completeness.is_complete(
    id="1",
    model="openai",
    rfp_url="https://example.com/rfp.pdf",
    ea_standard_eval_url="https://example.com/ea-standard.pdf",
    output_language="english"
)
```

## Troubleshooting

### Workers keep polling with no jobs

**Problem**: Workers continuously poll the database but find no jobs.

**Solutions**:
1. Check if there are jobs in the database with `need_to_check_completeness = '1'`
2. Enable idle timeout in `config.py` to auto-shutdown inactive workers
3. Use single job mode if you don't need database polling

### Email sending errors

**Problem**: "Connection unexpectedly closed" errors when sending emails.

**Solutions**:
1. Check SMTP credentials in `config.py`
2. Ensure you're using an app-specific password for Gmail
3. The system will retry sending emails automatically (3 attempts)

### Process doesn't exit

**Problem**: The process continues running after completing work.

**Solutions**:
1. Use Ctrl+C to gracefully shutdown
2. Enable `WORKER_IDLE_TIMEOUT_SECONDS` in config
3. Use single job mode for one-off processing

## Best Practices

1. **For production**: Use service mode with proper database job queue
2. **For testing**: Use single job mode with direct parameters
3. **For development**: Set a short idle timeout to avoid hanging processes
4. **Monitor logs**: Check the `logs/` directory for detailed execution logs 