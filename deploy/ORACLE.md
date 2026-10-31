# Deploying to Oracle Cloud Always Free

Result: `https://<your-ip-with-dashes>.sslip.io`, a read-only search demo, HTTPS, cost $0.
`sslip.io` is a free DNS service: `1-2-3-4.sslip.io` resolves to `1.2.3.4`, so you get a
real hostname (and a certificate) without buying a domain.

## 1. Create the VM (Oracle console, in your browser)
1. Sign up at cloud.oracle.com. Pick your **home region** carefully, it can't be changed. A card is needed for verification.
2. Compute → Instances → **Create instance**.
   - Image: **Ubuntu 24.04** (Canonical Ubuntu), shape: **Ampere (VM.Standard.A1.Flex)**, **2 OCPU, 12 GB RAM**.
   - "Out of capacity" is common: retry later or at another time of day.
   - Networking: keep the default VCN, **assign a public IPv4 address**.
   - Add your SSH public key (or let Oracle generate one and download it).
3. In the VCN's **security list**, add ingress rules for TCP ports **80** and **443** from `0.0.0.0/0`.

## 2. Prepare the VM (SSH in: `ssh ubuntu@<public-ip>`)
```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu && newgrp docker
# Ubuntu images on Oracle block ports with iptables as well as the security list:
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save
```

## 3. Deploy
```bash
git clone -b hardening https://github.com/sanjarmelikov/vector-search.git && cd vector-search
export DOMAIN=<ip-with-dashes>.sslip.io      # e.g. 129-146-10-20.sslip.io
docker compose -f deploy/docker-compose.prod.yml up -d --build
```
The image build downloads PyTorch (several minutes). On first start the service builds the
index in the background (several minutes on ARM); until then `/health` returns 503.
Watch it: `docker compose -f deploy/docker-compose.prod.yml logs -f api`.

## 4. Check
- `curl https://$DOMAIN/health` → 200 when the index is ready.
- Open `https://$DOMAIN/` and search.
- `curl -X POST https://$DOMAIN/documents ...` → 403 (read-only).

## Updating and stopping
- Update: `git pull && docker compose -f deploy/docker-compose.prod.yml up -d --build`
- Stop: `docker compose -f deploy/docker-compose.prod.yml down`

## Notes
- Oracle may reclaim idle Always Free instances (low CPU for 7 days): the demo gets used rarely, so open it now and then.
- `DOMAIN` must be exported in each new shell (or put it in a `.env` file next to the compose file).
