# Run BP Playwright Collection Daily at 6 AM

This sets Windows Task Scheduler to run:

```powershell
python collect_virtuals.py collect
```

every day at 6:00 AM.

## Steps

1. Press the `Windows` key.

2. Search for `Task Scheduler` and open it.

3. On the right side, click `Create Basic Task`.

4. Name the task:

```text
BP Daily Playwright Collect
```

5. Click `Next`.

6. Choose `Daily`, then click `Next`.

7. Set the time to:

```text
6:00:00 AM
```

8. Click `Next`.

9. Choose `Start a program`, then click `Next`.

10. In `Program/script`, enter:

```text
D:\Codex things\BP\.venv\Scripts\python.exe
```

11. In `Add arguments`, enter:

```text
collect_virtuals.py collect
```

12. In `Start in`, enter:

```text
D:\Codex things\BP
```

13. Click `Next`.

14. Click `Finish`.

## Test It

1. Open `Task Scheduler`.

2. Find `BP Daily Playwright Collect`.

3. Right-click it.

4. Click `Run`.

## Important

Do not put `(.venv)` or `(base)` in Task Scheduler. Those are only shown in PowerShell.

Task Scheduler should use this Python directly:

```text
D:\Codex things\BP\.venv\Scripts\python.exe
```
