terraform {
  required_version = ">= 1.5"
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.45"
    }
  }
}

variable "hcloud_token" {
  description = "Hetzner Cloud API token (or set HCLOUD_TOKEN env var)"
  type        = string
  default     = null
  sensitive   = true
}

provider "hcloud" {
  token = var.hcloud_token
}

resource "hcloud_ssh_key" "this" {
  name       = "${var.name}-key"
  public_key = var.ssh_public_key
}

resource "hcloud_firewall" "app" {
  name = "${var.name}-fw"

  # SSH
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = var.allowed_ssh_cidrs
  }

  # ACME HTTP-01 challenge (certbot). Nothing listens on 80 day-to-day —
  # docker-compose.yml never publishes it — but certbot needs it reachable
  # for the initial cert and every renewal.
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "80"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # HTTPS, published by the caddy service (docker-compose.yml).
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "443"
    source_ips = ["0.0.0.0/0", "::/0"]
  }
}

resource "hcloud_server" "app" {
  name        = var.name
  server_type = var.server_type
  # server_type/image architectures must match; see variables.tf -- this
  # defaults to arm64 (cax21), the closer match to the AWS Graviton2 box
  # it replaces, with x86 (cx33) available as an explicit override.
  image        = var.image
  location     = var.location
  ssh_keys     = [hcloud_ssh_key.this.id]
  firewall_ids = [hcloud_firewall.app.id]
  user_data    = file("${path.module}/cloud-init.yaml")

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }
}
