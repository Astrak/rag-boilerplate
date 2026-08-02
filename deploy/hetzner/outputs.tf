output "public_ip" {
  value = hcloud_server.app.ipv4_address
}

output "server_id" {
  value = hcloud_server.app.id
}
