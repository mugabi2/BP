# BP VPS Data Collection

This deployment runs the main virtual-results collection pipeline:

```text
betPawa public virtual-results pages -> Playwright -> PostgreSQL
```

No betPawa login or saved browser session is required for this collector.

The collector runs immediately when the stack starts. After a successful run it starts the next run on the configured interval (default: 3 hours). Failed runs retry sooner (default: 5 minutes).

## 1. Clone the deployment branch

```bash
sudo mkdir -p /opt/bp
sudo chown "$USER":"$USER" /opt/bp
git clone -b vps-data-collection https://github.com/mugabi2/BP.git /opt/bp
cd /opt/bp
```

If the repository is already cloned:

```bash
cd /opt/bp
git fetch origin
git checkout vps-data-collection
git pull
```

## 2. Create the VPS environment file

```bash
cp .env.example .env
nano .env
```

At minimum, replace:

```text
POSTGRES_PASSWORD=change_me
```

with a strong password. Do not commit the real `.env` file.

## 3. Start collection

```bash
docker compose up -d --build
```

PostgreSQL is intentionally not published on a host port. The collector reaches it through Docker's internal network.

## 4. Check status and logs

```bash
docker compose ps
docker compose logs -f collector
```

The collector should log that it started a collection immediately.

## 5. Verify PostgreSQL is receiving data

```bash
docker compose exec postgres sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT COUNT(*) AS seasons FROM seasons; SELECT COUNT(*) AS matches FROM match_results;"'
```

## Updating code later

After changes are committed to GitHub:

```bash
cd /opt/bp
git pull
docker compose up -d --build
```

## Stop without deleting data

```bash
docker compose down
```

Do not use `docker compose down -v` unless you intentionally want to delete the PostgreSQL volume.
