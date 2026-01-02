def generate_webvtt(subtitles):
    webvtt_content = "WEBVTT\n\n"
    for index, (start, end, text) in enumerate(subtitles):
        webvtt_content += f"{index + 1}\n"
        webvtt_content += f"{format_time(start)} --> {format_time(end)}\n"
        webvtt_content += f"{text}\n\n"
    return webvtt_content.strip()

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"

def save_webvtt_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)