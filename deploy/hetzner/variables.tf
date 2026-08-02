variable "name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "rag-boilerplate"
}

variable "location" {
  description = "Hetzner datacenter location"
  type        = string
  default     = "nbg1" # Nuremberg; also fsn1, hel1 (EU), ash/hil (US)
}

variable "server_type" {
  description = "Hetzner server type. cax21 = 4 vCPU ARM64 (Ampere Altra) / 8 GB (default -- same architecture family as the AWS r6g.medium (Graviton2) this replaces). cx33 = 4 vCPU x86 / 8 GB, an explicit override for anyone who wants x86 anyway (e.g. to sidestep arm64 wheel availability for some dependency) -- it is NOT more reliably orderable than cax21 despite what an earlier version of this comment assumed; a live `hcloud server-type list -o columns=name,location_available` check showed both empty across every location at the same time, so there's no current stock-based reason to prefer one over the other. Set this var and Hetzner resolves the matching-arch 'ubuntu-24.04' image automatically either way."
  type        = string
  default     = "cax21"
}

variable "image" {
  description = "OS image name. Hetzner resolves this to the arm64 or x86 build automatically based on server_type's architecture, so this doesn't need to change if you switch between cax* and cx* types."
  type        = string
  default     = "ubuntu-24.04"
}

variable "ssh_public_key" {
  description = "Public key material for SSH access (contents of e.g. ~/.ssh/id_ed25519.pub)"
  type        = string
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to reach port 22. Defaults to open (0.0.0.0/0) since this box is typically administered from a mobile/dynamic IP with no fixed CIDR to scope to -- security instead relies on SSH being key-only (Ubuntu's cloud image disables password auth by default, and cloud-init.yaml makes that explicit) plus fail2ban (also installed by cloud-init.yaml) to rate-limit brute-force attempts. Narrow this to your own IP/32 if you ever have a static one to scope to."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
