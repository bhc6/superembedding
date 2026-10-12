from scripts.export_general_rebuild_manifest import FIELDS,public_record


def test_public_manifest_uses_allowlist_and_omits_private_or_upstream_content():
    record={key:'value' for key in FIELDS}
    record.update(image='C:/private/image.jpg',texts=[{'user':'question','assistant':'answer'}],
                  download_url='https://example.com/signed?token=private',api_key='private')
    result=public_record(record)
    assert set(result)==set(FIELDS)
    assert 'private' not in str(result) and 'question' not in str(result)
