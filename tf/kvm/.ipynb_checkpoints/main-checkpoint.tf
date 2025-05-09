terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = ">= 1.47.0"
    }
  }
}

provider "openstack" {
  cloud = "openstack"
}

resource "openstack_compute_instance_v2" "node1" {
  name        = "node1-project10"
  flavor_name = "m1.large"                # large flavor
  image_name  = "CC-Ubuntu24.04"
  key_pair    = "id_rsa_chameleon"
  security_groups = ["default"]

  network {
    name = "sharednet1"
  }
}

resource "openstack_compute_instance_v2" "node2" {
  name        = "node2-project10"
  flavor_name = "m1.medium"               # medium flavor
  image_name  = "CC-Ubuntu24.04"
  key_pair    = "id_rsa_chameleon"
  security_groups = ["default"]

  network {
    name = "sharednet1"
  }
}

resource "openstack_compute_instance_v2" "node3" {
  name        = "node3-project10"
  flavor_name = "m1.medium"               # medium flavor
  image_name  = "CC-Ubuntu24.04"
  key_pair    = "id_rsa_chameleon"
  security_groups = ["default"]

  network {
    name = "sharednet1"
  }
}

