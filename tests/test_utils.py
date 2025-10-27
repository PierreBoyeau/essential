from essential.utils import _parse_gene_group


def test_parse_gene_group_single_gene():
    """Test: gene1 (single gene like 'nadE')"""
    assert _parse_gene_group("nadE") == ["nadE"]


def test_parse_gene_group_gene_group():
    """Test: gene1group (gene group like 'gspCDEFGHIJKLMO')"""
    result = _parse_gene_group("gspCDEFGHIJKLMO")
    expected = [
        "gspC",
        "gspD",
        "gspE",
        "gspF",
        "gspG",
        "gspH",
        "gspI",
        "gspJ",
        "gspK",
        "gspL",
        "gspM",
        "gspO",
    ]
    assert result == expected


def test_parse_gene_group_single_gene_hyphen_single_gene():
    """Test: gene1-gene2 (two single genes like 'uof-fur')"""
    # Note: _parse_gene_group only handles one group at a time
    # The hyphen splitting is done by _parse_target_genes
    assert _parse_gene_group("uof") == ["uof"]
    assert _parse_gene_group("fur") == ["fur"]


def test_parse_gene_group_gene_group_simple():
    """Test: gene1group-gene2 becomes gene1group (when tested individually)"""
    # Testing gene groups individually
    assert _parse_gene_group("sdhCDAB") == ["sdhC", "sdhD", "sdhA", "sdhB"]
    assert _parse_gene_group("sdhX") == ["sdhX"]


def test_parse_gene_group_multiple_gene_groups():
    """Test: gene1group-gene2group-gene3group (tested individually)"""
    # Each group tested separately
    assert _parse_gene_group("sdhCDAB") == ["sdhC", "sdhD", "sdhA", "sdhB"]
    assert _parse_gene_group("sucABCD") == ["sucA", "sucB", "sucC", "sucD"]
    assert _parse_gene_group("sdhX") == ["sdhX"]


def test_parse_gene_group_edge_cases():
    """Test edge cases"""
    # Two uppercase letters
    assert _parse_gene_group("hdeAB") == ["hdeA", "hdeB"]

    # Three uppercase letters
    assert _parse_gene_group("chiZPQ") == ["chiZ", "chiP", "chiQ"]

    # Empty string
    assert _parse_gene_group("") == []
