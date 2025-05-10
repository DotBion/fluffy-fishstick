mkdir -p /work/mlops-project10/tf/kvm
cd /work/mlops-project10/tf/kvm

mkdir -p /work/.local/bin
cd /work/.local/bin
wget -q https://releases.hashicorp.com/terraform/1.8.4/terraform_1.8.4_linux_amd64.zip
unzip -o terraform_1.8.4_linux_amd64.zip
rm terraform_1.8.4_linux_amd64.zip
export PATH="/work/.local/bin:$PATH"

cd /work/mlops-project10/tf/kvm

cat > main.tf <<EOF
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
  flavor_name = "m1.large"
  image_name  = "Ubuntu 24.04"
  key_pair    = "id_rsa_chameleon"
  security_groups = ["default"]

  network {
    name = "sharednet1"
  }
}

resource "openstack_compute_instance_v2" "node2" {
  name        = "node2-project10"
  flavor_name = "m1.medium"
  image_name  = "Ubuntu 24.04"
  key_pair    = "id_rsa_chameleon"
  security_groups = ["default"]

  network {
    name = "sharednet1"
  }
}

resource "openstack_compute_instance_v2" "node3" {
  name        = "node3-project10"
  flavor_name = "m1.medium"
  image_name  = "Ubuntu 24.04"
  key_pair    = "id_rsa_chameleon"
  security_groups = ["default"]

  network {
    name = "sharednet1"
  }
}
EOF

echo "/work/mlops-project10/tf/kvm/clouds.yaml is required."
read -p "Press enter once you've added clouds.yaml..."

export OS_CLIENT_CONFIG_FILE="/work/mlops-project10/tf/kvm/clouds.yaml"

cd /work/mlops-project10/tf/kvm
terraform init
terraform apply -auto-approve
