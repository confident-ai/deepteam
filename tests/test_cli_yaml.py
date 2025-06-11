from deepteam.cli import main as cli

sample = {
    "target": {"purpose": "t"},
    "default_vulnerabilities": [{"name": "Bias", "types": ["race"]}],
    "attacks": [{"name": "Prompt Injection", "weight": 1}],
}


def test_build_objects():
    vulns = [cli._build_vulnerability(v) for v in sample["default_vulnerabilities"]]
    attacks = [cli._build_attack(a) for a in sample["attacks"]]
    assert vulns[0].get_name() == "Bias"
    assert attacks[0].get_name() == "Prompt Injection"
