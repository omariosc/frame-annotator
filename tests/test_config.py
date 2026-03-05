import pytest
from frame_annotator.config import load_config, get_default_config, _validate


def test_default_config():
    config = get_default_config()
    assert config['project']['name'] == 'Frame Annotator'
    assert len(config['classes']) == 2
    assert config['classes'][0]['id'] == 'positive'
    assert config['classes'][1]['id'] == 'negative'


def test_validate_minimal():
    config = {
        'classes': [
            {'id': '0', 'name': 'A', 'color': '#000000'},
        ]
    }
    result = _validate(config)
    assert result['project']['name'] == 'Frame Annotator'
    assert result['images']['pattern'] == '*.png'


def test_validate_rejects_no_classes():
    with pytest.raises(ValueError, match='at least one class'):
        _validate({'classes': []})


def test_validate_rejects_missing_fields():
    with pytest.raises(ValueError, match='missing required field'):
        _validate({'classes': [{'id': '0', 'name': 'A'}]})


def test_validate_rejects_duplicate_ids():
    with pytest.raises(ValueError, match='Duplicate class id'):
        _validate({'classes': [
            {'id': '0', 'name': 'A', 'color': '#000000'},
            {'id': '0', 'name': 'B', 'color': '#111111'},
        ]})


def test_validate_rejects_invalid_color():
    with pytest.raises(ValueError, match='Invalid hex color'):
        _validate({'classes': [
            {'id': '0', 'name': 'A', 'color': 'red'},
        ]})


def test_validate_rejects_duplicate_shortcuts():
    with pytest.raises(ValueError, match='Duplicate shortcut'):
        _validate({'classes': [
            {'id': '0', 'name': 'A', 'color': '#000000', 'shortcut': '1'},
            {'id': '1', 'name': 'B', 'color': '#111111', 'shortcut': '1'},
        ]})


def test_validate_subcategories():
    config = {
        'classes': [
            {'id': '0', 'name': 'Safe', 'color': '#28a745'},
            {'id': '1', 'name': 'Unsafe', 'color': '#dc3545',
             'subcategories': [
                 {'id': 'a', 'name': 'Posture', 'shortcut': 'a'},
                 {'id': 'b', 'name': 'Extension', 'shortcut': 'b'},
             ]},
        ]
    }
    result = _validate(config)
    assert len(result['classes'][1]['subcategories']) == 2


def test_validate_rejects_long_subcategory_id():
    with pytest.raises(ValueError, match='single character'):
        _validate({'classes': [
            {'id': '1', 'name': 'X', 'color': '#000000',
             'subcategories': [{'id': 'ab', 'name': 'Y'}]},
        ]})
