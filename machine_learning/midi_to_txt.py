import glob
from music21 import converter, instrument, note, chord

def midi_to_text(file_path):
	notes = []
	for file in glob.glob(file_path):
		midi = converter.parse(file)
		notes_to_parse = None
		notes_to_parse = midi.flat.notes
		for element in notes_to_parse:
			if isinstance(element, note.Note):
				notes.append(str(element.pitch))
			elif isinstance(element, chord.Chord):
				notes.append('.'.join(str(n) for n in element.normalOrder))
	return(notes)


def text_to_midi(file_path):
	return(midi_file)

out = midi_to_text("data/scarlatti_midi/*.MID")
with open('data/training_data.pkl', 'wb') as f:
    pickle.dump([out], f)
print("done")


