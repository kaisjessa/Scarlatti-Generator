#imports
import glob
import pickle
import random
from music21 import converter, instrument, note, chord

#returns character sequence from midi folder
def midi_to_text(file_path, num_files=555):
	#array of characters representing notes
	notes = []
	#go through all midi files in folder
	i = 1
	for file in glob.glob(file_path):
		if(i > num_files):
			return(notes)
		i += 1
		#read the file
		midi = converter.parse(file)
		#current notes
		notes_to_parse = None
		notes_to_parse = midi.flat
		#print([e for e in midi.flat])
		#add notes and chords to array
		for element in notes_to_parse:
			if isinstance(element, note.Note):
				notes.append(str(element.pitch))
			elif isinstance(element, chord.Chord):
				notes.append('.'.join(str(n) for n in element.normalOrder))
			elif isinstance(element, note.Rest):
				notes.append(" rest")
		print("Parsing %s" % file)
	#return array
	return(notes)

out = midi_to_text("data/scarlatti_d_minor_sonatas/*.MID", 32)
print(len(out))
with open('data/models/d_minor_training_data.pkl', 'wb') as f:
    pickle.dump([out], f)
print("done")