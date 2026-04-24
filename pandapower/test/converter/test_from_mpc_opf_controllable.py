from pathlib import Path

from pandapower.converter.matpower import from_mpc


def test_from_mpc_opf_controllable(tmp_path: Path):
    mpc_file = tmp_path / "case_from_mpc_opf_controllable.m"
    mpc_file.write_text(
        """\
        function mpc = case_from_mpc_opf_controllable
        mpc.version = '2';
        mpc.baseMVA = 100.0;
        mpc.bus = [
            1 3 0 0 0 0 1 1.0 0.0 110.0 1 1.1 0.9;
            2 2 0 0 0 0 1 1.0 0.0 110.0 1 1.1 0.9;
        ];
        mpc.branch = [
            1 2 0.01 0.05 0 100 100 100 0 0 1 -30 30;
        ];
        mpc.gen = [
            1 50 0 100 -100 1.0 100 1 100 0;
            2 20 0 100 -100 1.0 100 1 100 0;
        ];
        mpc.gencost = [
            2 0 0 2 1 0;
            2 0 0 2 1 0;
        ];
        end
        """,
        encoding="utf-8",
        )

    net = from_mpc(
        str(mpc_file),
        validate_conversion=False,
        set_opf_controllable=True,
    )

    assert net.ext_grid["controllable"].all()
    assert net.gen["controllable"].all()