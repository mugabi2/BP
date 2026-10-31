# BP VPS Data Collection

This deployment runs the main virtual-results collection pipeline:

```text
betPawa public virtual-results pages -> Playwright -> PostgreSQL
```

No betPawa login or saved browser session is required for this collector.

The collector runs once per day at 1:00 PM Africa/Kampala time. Successful runs then stay idle until the next day's 1:00 PM schedule. Failed runs retry after 5 minutes.

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

The collector should log its next scheduled collection time. At 1:00 PM Kampala time it starts the collection.

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


## Run a harvest manually

You do not have to wait for the 1:00 PM schedule. Trigger a harvest at any time with:

```bash
cd /opt/bp
docker compose exec collector python collect_virtuals.py collect
```

The scheduled run and manual run share a PostgreSQL advisory lock. If one harvest is already active, a second trigger exits cleanly instead of running two collectors at once.

Incomplete matchdays or seasons do not stop the overall harvest. The collector continues to later matchdays and seasons, leaves the incomplete season marked unsuccessful, and can retry its missing data on a future run.


## Metabase

The Docker stack also runs Metabase on port 3000 by default.

Start or update the stack:

```bash
cd /opt/bp
git pull
docker compose up -d --build
```

Check the services:

```bash
docker compose ps
docker compose logs -f metabase
```

Open Metabase in a browser:

```text
http://YOUR_VPS_IP:3000
```

Complete the first-run Metabase admin setup, then add the BetPawa PostgreSQL database with:

```text
Database type: PostgreSQL
Display name: BP
Host: postgres
Port: 5432
Database name: bp
Username: bp_user
Password: the POSTGRES_PASSWORD value from .env
SSL: off
```

Metabase stores its own users, questions, and dashboards in a separate PostgreSQL database named `metabase_app`. BetPawa match data remains in `bp`.

If port 3000 is blocked by the VPS firewall, allow TCP 3000 temporarily and then reload the page. Once Metabase is working, prefer putting it behind HTTPS/reverse proxy rather than leaving a raw public port as the permanent setup.
