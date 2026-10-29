from betterbench.env import pci_gpus


def _card(root, name, addr, vendor, resource, link=("16.0 GT/s PCIe", "16")):
    dev = root / "devices" / addr
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    (dev / "device").write_text("0xe223\n")
    (dev / "resource").write_text(resource)
    (dev / "current_link_speed").write_text(link[0] + "\n")
    (dev / "current_link_width").write_text(link[1] + "\n")
    card = root / "drm" / name
    card.mkdir(parents=True)
    (card / "device").symlink_to(dev)


def test_intel_rebar_on(tmp_path):
    # BAR0 16 MiB registers, BAR2 32 GiB VRAM aperture: Resizable BAR on
    _card(tmp_path, "card0", "0000:03:00.0", "0x8086",
          "0x0000007800000000 0x0000007800ffffff 0x0\n"
          "0x0000000000000000 0x0000000000000000 0x0\n"
          "0x0000007000000000 0x00000077ffffffff 0x0\n")
    (g,) = pci_gpus(str(tmp_path / "drm"))
    assert g["vendor"] == "intel" and g["pci_address"] == "0000:03:00.0"
    assert g["largest_bar_mib"] == 32768 and g["resizable_bar"] == "likely on"
    assert g["current_link_speed"] == "16.0 GT/s PCIe" and g["current_link_width"] == "16"


def test_rebar_off_256mib(tmp_path):
    _card(tmp_path, "card0", "0000:01:00.0", "0x1002",
          "0x00000000e0000000 0x00000000efffffff 0x0\n")
    (g,) = pci_gpus(str(tmp_path / "drm"))
    assert g["vendor"] == "amd" and g["largest_bar_mib"] == 256 and g["resizable_bar"] == "likely off"


def test_skips_non_gpu_vendor_and_duplicate_nodes(tmp_path):
    _card(tmp_path, "card0", "0000:03:00.0", "0x8086", "0x0 0xfffffff 0x0\n")
    (tmp_path / "drm" / "card0-DP-1").mkdir()  # connector node, no device link
    (tmp_path / "drm" / "card1").mkdir()
    (tmp_path / "drm" / "card1" / "device").symlink_to(tmp_path / "devices" / "0000:03:00.0")  # same GPU again
    _card(tmp_path, "card2", "0000:00:02.0", "0x1234", "0x0 0xfff 0x0\n")
    gpus = pci_gpus(str(tmp_path / "drm"))
    assert [g["pci_address"] for g in gpus] == ["0000:03:00.0"]


def test_link_read_at_card_switch_upstream_port(tmp_path):
    # Arc: root port -> Intel switch upstream port -> Intel downstream port -> GPU.
    # The GPU function only sees the internal link; the slot link is at the upstream port.
    root = tmp_path / "devices" / "0000:00:01.0"
    up = root / "0000:01:00.0"
    down = up / "0000:02:01.0"
    gpu = down / "0000:03:00.0"
    gpu.mkdir(parents=True)
    for d, vendor, cls, link in ((root, "0x8086", "0x060400", ("16.0 GT/s PCIe", "16")),
                                 (up, "0x8086", "0x060400", ("16.0 GT/s PCIe", "8")),
                                 (down, "0x8086", "0x060400", ("2.5 GT/s PCIe", "1")),
                                 (gpu, "0x8086", "0x030000", ("2.5 GT/s PCIe", "1"))):
        (d / "vendor").write_text(vendor + "\n")
        (d / "class").write_text(cls + "\n")
        (d / "current_link_speed").write_text(link[0] + "\n")
        (d / "current_link_width").write_text(link[1] + "\n")
    (gpu / "device").write_text("0xe223\n")
    (gpu / "resource").write_text("0x0000007000000000 0x00000077ffffffff 0x0\n")
    card = tmp_path / "drm" / "card0"
    card.mkdir(parents=True)
    (card / "device").symlink_to(gpu)
    (g,) = pci_gpus(str(tmp_path / "drm"))
    # on an Intel host the root port is an Intel bridge too; either end of the slot link reads the same link
    assert g["current_link_speed"] == "16.0 GT/s PCIe"
    assert g["pcie_link_port"] in ("0000:01:00.0", "0000:00:01.0")


def test_vm_root_port_is_not_climbed(tmp_path):
    # VM passthrough: the GPU hangs off a QEMU root port (vendor 0x1b36); keep the GPU's own link
    root = tmp_path / "devices" / "0000:00:1c.2"
    gpu = root / "0000:03:00.0"
    gpu.mkdir(parents=True)
    (root / "vendor").write_text("0x1b36\n")
    (root / "class").write_text("0x060400\n")
    for f, v in (("vendor", "0x8086"), ("class", "0x030000"), ("device", "0xe223"),
                 ("current_link_speed", "2.5 GT/s PCIe"), ("current_link_width", "1"),
                 ("resource", "0x0 0xfffffff 0x0")):
        (gpu / f).write_text(v + "\n")
    card = tmp_path / "drm" / "card0"
    card.mkdir(parents=True)
    (card / "device").symlink_to(gpu)
    (g,) = pci_gpus(str(tmp_path / "drm"))
    assert g["current_link_speed"] == "2.5 GT/s PCIe" and "pcie_link_port" not in g


def test_missing_sysfs_is_empty(tmp_path):
    assert pci_gpus(str(tmp_path / "nope")) == []
