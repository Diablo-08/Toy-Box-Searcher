try:
    from google import genai
    print('import success')
except Exception as e:
    print('import failed', e)
