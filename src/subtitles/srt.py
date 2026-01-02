def create_srt_subtitle(content, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for index, (start, end, text) in enumerate(content):
            f.write(f"{index + 1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def parse_srt_file(file_path):
    subtitles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    index = 0
    while index < len(lines):
        if lines[index].strip().isdigit():
            index += 1
            time_line = lines[index].strip()
            start, end = time_line.split(' --> ')
            start = convert_time_to_seconds(start)
            end = convert_time_to_seconds(end)
            index += 1
            text = lines[index].strip()
            subtitles.append((start, end, text))
            index += 2  # Move to the next subtitle block
        else:
            index += 1
    return subtitles

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000