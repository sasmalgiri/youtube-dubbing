def convert_timecode_to_seconds(timecode):
    hours, minutes, seconds = map(float, timecode.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def convert_seconds_to_timecode(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}"