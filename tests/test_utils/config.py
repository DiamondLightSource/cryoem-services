from __future__ import annotations

import os


def cluster_submission_configuration(tmp_path):
    # Create a config file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as cf:
        cf.write("rabbitmq_credentials: rmq_creds\n")
        cf.write(f"recipe_directory: {tmp_path}/recipes\n")
        cf.write("slurm_credentials:\n")
        cf.write(f"  default: {tmp_path}/slurm_credentials.yaml\n")
        cf.write(f"  extra: {tmp_path}/slurm_credentials_extra.yaml\n")
    os.environ["USER"] = "user"

    # Create dummy slurm credentials files
    with open(tmp_path / "slurm_credentials.yaml", "w") as slurm_creds:
        slurm_creds.write(
            "user: user\n"
            "user_home: /home\n"
            f"user_token: {tmp_path}/token.txt\n"
            "required_directories: [directory1, directory2]\n"
            "partition: partition\n"
            "partition_preference: preference\n"
            "cluster: cluster\n"
            "url: /url/of/slurm/restapi\n"
            "api_version: v0.0.40\n"
        )
    with open(tmp_path / "slurm_credentials_extra.yaml", "w") as slurm_creds:
        slurm_creds.write("url: /slurm/extra/url\n")
        slurm_creds.write("api_version: v0.0.41\n")
        slurm_creds.write("user: user2\n")
        slurm_creds.write(f"user_token: {tmp_path}/token_user2.txt\n")
        slurm_creds.write("partition: part\n")
        slurm_creds.write("partition_preference: preference\n")

    with open(tmp_path / "token.txt", "w") as token_file:
        token_file.write("token_key")
    with open(tmp_path / "token_user2.txt", "w") as token_file:
        token_file.write("token_key2")
