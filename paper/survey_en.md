# Three Tokenization Representations for Deep Symbolic Music Generation, and Their Visualization

Working English translation of `survey_zh(1).md`. Not a submission draft.


## Abstract

Before symbolic music enters a generative model, a representation has to be chosen. This paper is not a general survey of music generation. It discusses tokenization for symbolic music, classified by how coarse an object the model actually accepts, and it visualizes the same monophonic excerpt rewritten as each of the three objects. In the first class, one step is still a note or one attribute of a note; event sequences and ABC notation belong here. The second class writes music as a two-dimensional table aligned in time and pitch, or in time and track, with one or more channels; the model denoises that table or a continuous latent derived from it. In the third class, one step is a learned code covering a span longer than one note, and the code itself has no pitch name. The learning task follows the object: next-symbol prediction or masked diffusion on note-level symbols, two-dimensional denoising or masked prediction on a roll, and sequence prediction on coarse codes. The figures place the three objects on one score. Coarser units that follow musical meaning are what the third class has begun to do, and they are the direction worth pursuing further.

Keywords: symbolic music, tokenization, piano roll, diffusion models, residual quantization, ABC notation


## 1. Introduction

Existing surveys organize the field along different axes. Ji et al. divide deep music generation into score, performance, and audio: a score is a discrete notation of pitch and duration, a performance adds velocity and small timing deviations, and audio is the continuous waveform [Ji et al., 2020]. This paper discusses only symbolic music, but the same distinction remains inside the first class: MIDI-like records performance actions, while REMI and ABC stay closer to the score. The time grid is fixed in Section 2, not inside each class. Le et al. split symbolic sequences into time-slice and event schemes, and events further into atomic and compound tokens; placing several attributes of one note in successive tokens adds an artificial order beyond musical time [Le et al., 2025]. Li et al. list events, piano rolls, sequences, text, and graphs as storage forms [Li et al., 2025]. Ma et al., in a survey of music foundation models, separate hand-designed vocabularies from learned ones, and note that waveform, MIDI, and ABC are not yet covered by one foundation model [Ma et al., 2024].

Those classifications follow the representation itself or the modality. This paper follows that work but organizes schemes by the object the model actually accepts, in three classes:

- Class 1: note-level event sequences. One step is a note or one attribute of a note. Preprocessing only quantizes and orders the events. Next-symbol prediction and masked diffusion both sit here.
- Class 2: the piano roll as an image. Convolutional networks and diffusion models denoise or inpaint the two-dimensional table, or a continuous latent of that table.
- Class 3: coarse discrete codes. A code does not name a pitch. The downstream model predicts codebook indices, and a decoder reconstructs MIDI.
The three classes are compared on one excerpt: two bars, one track, 4/4, 120 BPM, six notes. In Figures 1 and 3–5 the notes stay fixed and the object changes. Figure 2 only delays the onset of E4 by 30 ms. Figure 1 is the sixteenth-note piano roll used as the reference. Figure 2, in Section 2, shows that 30 ms onset before and after it is quantized onto the beat grid. Figure 3 writes the excerpt as Class-1 event symbols and compares sequence lengths. Figure 4 writes it as a multi-channel roll, still on sixteenth notes. Figure 5 writes it as a GETScore, with tracks on the vertical axis. Class-3 codes cover spans longer than this excerpt, so the counts in Figure 6 are taken from the original papers, not encoded from this score. The number of tokens in a piece depends on its length, so totals from different pieces are not compared. This paper compares only how many steps this excerpt takes under each writing. The time grid is fixed in Section 2. Within Class 1, on that same grid, the comparison is how notes are packed: one attribute per step, or several attributes in one step.

ABC is placed in Class 1. Its characters are pitch names, durations, or bar lines, the same note-level facts as pitch, duration, and bar lines in REMI, carried as notation text. Byte-pair encoding is also in Class 1: it is stacked on an existing event vocabulary, merges frequent spans, and expands back to those note-level events. A Class-3 step is a quantized code covering more than one note. The code has no pitch name and must be reconstructed over that longer span.

Figure 1. Piano roll of the reference excerpt. 4/4, 120 BPM, two bars; each column is one sixteenth note. C4 and E4 start together, then G4, A4, and B4; beat 4 of bar 1 is a rest, and bar 2 is C5. Velocity is not marked. Figures 3–5 use the same notes. Figure 2 only delays the onset of E4.


## 2. Absolute-time grids and beat grids

An absolute-time grid and a beat grid are two ways of aligning musical content to a time axis. Each tokenization scheme in this paper first chooses how wide the smallest cell is: an absolute duration, such as 10 ms, or a beat on the score. Counts of steps or columns are not comparable across different cells. Class-3 codes do not write milliseconds or beat positions directly; before quantization, the span still falls on one of the two grids below.

The first grid divides time in absolute units. MAESTRO’s MIDI was captured by key sensors on a Disklavier during a competition, about 200 hours. Note times come from the sensors, not from audio transcription [Hawthorne et al., 2019]. Music Transformer uses 10 ms time-shift bins on this clock, from 10 ms to 1 s, written only where an event occurs [Huang et al., 2019]. GiantMIDI-Piano transcribes 10,855 classical piano recordings, 1,237 hours, about nine-tenths from live performance. Onsets, offsets, velocity, and pedal are estimates of the transcription model; the files have no beats, time signatures, or scores [Kong et al., 2022]. ATEPP is also transcribed from recordings: 11,742 performances, about 1,000 hours, and about half of the pieces also have MusicXML scores. The output is still MIDI with millisecond timing; the note information is estimated by a model, not captured by key sensors [Zhang et al., 2022].

The second grid divides time by beats. Once beats are aligned, an onset that falls off the beat is moved into a cell by the grid and the chosen rounding rule. POP909 contains 909 pop-piano arrangements with beats, chords, and modes marked. Its finest beat grid is four cells per beat, or sixteenth notes; under a 4/4 signature the two choices are the same. Tempos in POP909 are those marked for the pieces [Wang et al., 2020]. TheoryTab arranges 10,148 pop melodies with chords, also on a beat grid [Yeh, n.d.]. PDMX collects more than 250,000 public-domain MusicXML files from MuseScore; note positions are beat positions on the score [Long et al., 2024]. MusicXML writes pitch names, durations, and time signatures by bar. The models discussed here do not read that score format directly.

Lakh MIDI is often used for large-scale symbolic training. Release v0.1 has 176,581 unique files, scraped from the public web and deduplicated by checksum. Of these, 45,129 match a recording in the Million Song Dataset [Raffel, 2016]. The notes come from MIDI files already on the web. A few files may originally have been someone else’s notation. Timing inside the collection is not uniform: some files are already quantized to a beat grid, some keep a tempo curve, and some have no tempo mark and remain on absolute time. The collection as a whole is not one grid.

Turning milliseconds into beats requires beat positions. A tempo written in the file, beats tracked on a recording, beats aligned from a score, or beats tapped by hand can all be used to estimate the tempo, and all of them can be wrong. When the file states a BPM, one beat is 60/BPM seconds. With N cells per beat, the continuous grid position of a time t in seconds is u = N · BPM · t / 60.

The continuous position u is then rounded down or rounded to the nearest integer. If the BPM matches the beats and t = 0 falls on a beat, the absolute time of beat k is t = k · 60/BPM. Substitution gives u = Nk, an integer, exactly on beat k. A note that is not on a grid line has a non-integer u.

In 4/4, N = 4 is the sixteenth note. MIDI does not store seconds; t in the formula is computed from ticks using the tempo written in the file. If the tempo changes mid-file, the new tempo must be used. If the performance slows down while the file keeps a single tempo mark, the computed u in the ritardando runs ahead of the actual beat.

When a score exists, its notes are already written on beats, and a MIDI export of MusicXML is already on a beat grid. ASAP supplies both the score and the performance: the score stays on beats, and it also marks which score beat falls at which millisecond of the performance. The performance MIDI itself remains on an absolute-time grid [Foscarin et al., 2020].

Once performance milliseconds are placed in beat cells, the exact absolute time is gone. Writing the result back as a performance requires a separate duration for each cell. Figure 2 is fixed at 120 BPM and four cells per beat, so one cell is 125 ms. Before quantization, E4 is 30 ms late. Because 30 ms is less than half a cell, 62.5 ms, both floor and round place it on the beat, and the 30 ms is lost.

Figure 2. Onsets quantized to a sixteenth-note grid. At 120 BPM one cell is 125 ms and half a cell is 62.5 ms. (a) Before quantization, C4 is at 0 ms and E4 at 30 ms. (b) After quantization both are at 0 ms. The dashed line is the original 30 ms of E4.


## 3. Class 1: note-level event sequences

One step in this class is a note, or one attribute of a note: pitch, duration, velocity, beat position, bar line, or a character in ABC. The symbols form one total order. Preprocessing reads the MIDI file, quantizes it, and orders the events. No encoder is trained. A piece often becomes several thousand steps. Class 3 is also usually predicted by a deep sequence model, but one step there is a code with no pitch name. The boundary is this: after the symbols are expanded, they can still be read as note-level facts.

The learning task is not fixed by the class. The common task is autoregressive next-symbol prediction, usually with a decoder-only Transformer, or with Mamba and hybrid backbones. The same sequence can also be used for masked diffusion: symbols are replaced by [MASK], and the model predicts the original symbol. Positions that have been opened stay fixed. The schedule decides how many positions open at each step.

The grid was chosen in Section 2. MIDI-like and TSD use an absolute-time grid. REMI uses a beat grid.


### 3.1 MIDI-like

Oore et al. record the piano with note-on, note-off, velocity, and time shift [Oore et al., 2020]. Music Transformer uses that vocabulary [Huang et al., 2019]. Notes that sound together are written as successive note-ons, with no time shift between them. The vocabulary describes one piano track. Velocity and millisecond timing are kept. Time in the sequence is absolute time, not a beat, and meter is not in the vocabulary. Music Transformer uses relative attention to represent relative distances between events in a long sequence. A note-off must be paired with an earlier note-on. After training, the model can still emit an unpaired note-off, or a repeated note-on on the same pitch. Such a sequence cannot be turned back into playable MIDI reliably.


### 3.2 REMI and TSD

Both REMI and TSD replace note-off with a duration, so the model does not have to match a note-off to an earlier note-on. They differ in the clock.

Pop Music Transformer still uses a Transformer-XL backbone; what changes is the vocabulary [Huang and Yang, 2020]. REMI replaces time shift with Bar and Position: one Bar per bar, and 16 positions inside the bar. Notes that share a position share that Position token, and each note still has its own pitch, velocity, and duration. Both the onset and the offset are snapped to these 16 positions. Duration is counted in thirty-second notes, so one sixteenth note is written as duration 2. Meter is in the vocabulary, and the rhythm is steadier than in MIDI-like. Tempo is a separate token and may change during the piece. Positions are set at 16 per bar, so a mid-piece change of time signature, such as 4/4 to 3/4, is not represented. The number of positions in a REMI bar is fixed.

TSD also writes a note as pitch, velocity, and duration, but time remains a time shift. It does not write bars or beat positions [Fradet et al., 2023]. Meter has to be learned from those shifts.

Both writings use one attribute per step, so the sequence still grows with the number of attributes. A vocabulary can add events for a repertoire. MusicMamba, aimed at Chinese traditional melody, adds note-type and mode events to REMI and calls the result REMI-M [Chen et al., 2025]. Note type separates modal tones from passing tones. Mode events mark the start, end, and type of a mode. The prediction is still the next event. One Mamba branch handles melody and mode, a Transformer branch handles the global context, and a bidirectional Mamba merges them. The grain of the vocabulary does not change.


### 3.3 Packing several events into a note

When each step writes only one attribute, a song becomes thousands of steps. Compound word starts from REMI and places adjacent tokens that jointly define one event into the same step [Hsiao et al., 2021]. A note step fills only pitch, duration, and velocity. Position, chord, and tempo form another step. Every step outputs the same set of fields, and unused fields are [ignore]: position, chord, and tempo are empty in a note step, and pitch, duration, and velocity are empty in a metric step. A family token says which kind of step it is. Each field has its own head. The heads sample in parallel, so one head does not see the value just sampled by another. The sequence is counted in events. This representation is used to generate full pop-piano pieces.

Octuple also puts several fields in one step, but one step is one note. Bar, position, and tempo are written inside that note and do not take a step of their own [Zeng et al., 2021]. The eight fields are time signature, tempo, bar, position, instrument, pitch, duration, and velocity, concatenated into one vector. Because the time signature is a field, a change during the piece can be kept. The original paper uses it for masked pretraining and understanding. The length is about one quarter of REMI and one half of compound word.

Compound word and Octuple gather several steps into one. Byte-pair encoding is stacked on these event vocabularies and likewise merges adjacent tokens into one symbol, without starting a new representation [Fradet et al., 2023]. It counts the most frequent adjacent spans in an existing token stream and merges them, so the sequence is shorter. The merge table is reversible. Expanded symbols are still pitch, duration, or position. No encoder is trained, and there is no reconstruction loss. A new symbol may cross several events. Its boundary follows co-occurrence, not bars or phrases.


### 3.4 ABC notation

ABC writes a score as a line of text. With the eighth note as the unit length, Figure 1 is [CE]2 G2 AB z2 | c2 z6 |. Notes inside brackets sound together, and a number after a letter is a duration multiplier [Walshaw, 2011]. ChatMusician does not build a music vocabulary. It gives the whole string to the LLaMA tokenizer and still predicts the next token, so it stays in Class 1 [Yuan et al., 2024].

ChatMusician uses ABC for folk and traditional music, and the body of the string is the melody [Yuan et al., 2024]. Chord names sit in quotation marks to the left of a note. They contain a note name, an optional accidental, a chord type, and an optional bass. They do not say which pitches are used or in what rhythm. A player may generate accompaniment from the chord name. The texture is the player’s and is not in the ABC string [Walshaw, 2011].

Pitch names, durations, and simultaneous notes in brackets expand to note-level events, so they stay in this class. A chord name does not expand to a unique voicing. Velocity, milliseconds, and pedal are omitted by default. The tokenizer cuts subwords, so one token can fall in the middle of a note. Text and score share one model, so ChatMusician can use a text language model to generate symbolic music in ABC and to cover downstream tasks such as chord arrangement and music understanding. ABC still cannot preserve, without loss, the complex textures, polyphony, and dynamics of classical piano that are present in MIDI.

On the same songs written as MIDI, ABC uses fewer tokens. Yuan et al. counted 1,000 songs from the training set. ABC with the LLaMA tokenizer averages 288.21 tokens per song and 5.16 per second. The same songs as MIDI-like average 728.60 tokens per song and 12.42 per second, about 288 against 729. Counted as REMI, they average 753.41 tokens per song and 12.84 per second [Yuan et al., 2024]. Repeats are written once with |: and :|, so the string is shorter. LLaMA then cuts subwords, and one token may hold several characters, whereas MIDI-like and REMI take one step per event. Figure 3 counts event steps of the Figure 1 excerpt. The averages above are means over 1,000 full songs.

Figure 3. Figure 1 written as Class-1 symbols. (a) REMI. (b) ABC, with the eighth note as the unit. (c) Sequence length of each event scheme. The dashed line is the note count, 6. MIDI-like, REMI, and TSD use one attribute per step and exceed 6 steps. Compound word and Octuple put several attributes in one step, so the count falls. The numbers belong only to this beat-aligned excerpt.


### 3.5 Models that accept these symbols

The representation decides what a token contains. The model decides how that information is read. The two can be set side by side briefly.

Autoregressive symbol prediction moves left to right. Each step adds one symbol, and the context is the prefix already written. Music Transformer and Pop Music Transformer use Transformers. MusicMamba splits Mamba and Transformer into two branches and still predicts the next event.

Masked diffusion sees the whole sequence at once. The network predicts the original symbol at positions that are still masked. Each step opens only some of them, and positions already opened stay fixed. SCHmUBERT uses a Transformer as the denoiser. In 4/4 there are 16 steps per bar, and the input to masked diffusion is a one-dimensional index sequence laid out in time. Sixty-four bars of melody are 1,024 integers. Steps with no new event remain in the sequence, and pitch is written inside the integer. The model receives that one sequence. There is no time-by-pitch axis [Plasser et al., 2023]. SMDIM’s symbols are REMI. Its denoising block is called MFA: each block is one Mamba layer, one feed-forward layer, and one self-attention layer, and the model stacks eight blocks [Xu et al., 2026]. The authors discuss putting only one attention layer in each block together with the sparse attention placement used in hybrid models such as Jamba. The diffusion schedule decides how many positions open at a step. Which of the still-masked positions open is drawn separately. Both models still output note-level symbols, so they stay in this class.

The original MIDI-like vocabulary of Section 3.1 and the original REMI vocabulary of Section 3.2 record notes on a piano. If a multi-instrument file is written with those vocabularies, notes from every track are placed on one timeline and the instrument is not written. MidiTok is a package that implements MIDI-like, REMI, TSD, compound word, and Octuple. When several tracks are needed, a Program token can be added to the vocabulary. The instrument then stays in the same sequence, and the file does not have to be split by track before tokenization [Fradet et al., 2021]. MuMIDI places a Track token before a group of notes. The track types are melody, drum, piano, strings, guitar, and bass. Another Track token is written when the track changes, and the downstream model still predicts the next symbol [Ren et al., 2020]. In Octuple, one step is one note and the instrument is one of eight fields: 0–127 are ordinary instruments and 128 is percussion. In the paper’s example, piano and guitar notes are ordered by time in one sequence [Zeng et al., 2021]. Unconditional compound-word generation writes only piano, with no track mark. Conditional generation adds a track mark to separate a given lead sheet from the piano to be generated [Hsiao et al., 2021].


## 4. Class 2: the piano roll as an image

Class 2 writes music as a table of pitch against time. The horizontal axis is the time grid and the vertical axis is pitch. Notes that sound together occupy the same column, so the model does not pair a note-on with a note-off. The excerpt in Figure 4 is one piano track. Several instruments can be one roll each, or the track itself can be an axis of the table. A piano roll is a sparse two-dimensional image.

One channel is not enough. If a single binary channel only marks whether a pitch is sounding, two adjacent notes on the same pitch join into one long note, and the onset cannot be seen. A binary piano-roll image is therefore split into at least two channels: one for onset and one for sustain. Polyffusion and Proffusion use that pair. Figure 4(a) draws sustain and onset on one image. Panels (b)–(d) separate velocity, onset, and pedal, to show how channels can be divided. The three channels used by Huang et al. on performance data are the roll frame, onset, and pedal. That set is not the same as the velocity channel in Figure 4 [Huang et al., 2024].

Early models generate a few bars directly on this table. MidiNet uses convolution to generate melody. MuseGAN places one roll each for bass, drums, guitar, piano, and strings, and generates the channels together [Yang et al., 2017; Dong et al., 2018]. Coconet repeatedly rewrites masked positions on a four-voice grid [Huang et al., 2017]. The canvas is usually only a few bars. Because a piano roll is sparse, most cells are 0.

Figure 4. The two-dimensional roll of Figure 1. The horizontal axis is sixteenth notes. (a) Green is sustain and red is onset. (b) Velocity: C4 and E4 are 80 and are drawn lighter; the others are 90. (c) Onset, occupying only the first cell. (d) Pedal, which is entirely 0 in this excerpt.


### 4.2 Denoising and reading the result back as MIDI

When denoising is done directly on the roll, the forward process adds Gaussian noise to the cell values and the network predicts the denoised table. Discrete diffusion in Class 1 has a different object: the sequence is symbols such as REMI, the noise replaces a symbol with a mask, and the network predicts the original vocabulary symbol. Once it is opened, it is a discrete event. Here the network still outputs a continuous value in each cell.

Continuous values have to be turned into notes before they can be written as MIDI. Binary channels are thresholded to 0 or 1: the onset channel gives the start, and the sustain channel gives which cells the note occupies. If the two channels disagree, the events do not line up. Sustain without onset has no start and is dropped. If sustain breaks after an onset, one note is cut into several. Adjacent notes of the same pitch join into one longer note if the onset channel does not mark a new onset. Velocity or pedal channels likewise have to be reduced to levels that MIDI can store. The diffusion steps themselves do not check these relations.

The object accepted by Polyffusion is a roll of fixed length, eight bars at a time [Min et al., 2023]. In 4/4 with four cells per beat, eight bars are 32 beats and the time axis has 128 cells. Pitch uses MIDI 0 through 127, another 128 cells. Onset and sustain are two channels, with shape 2×128×128. The model denoises this image. In unconditional sampling the roll starts as pure noise, and the end of denoising is these eight bars.

Pieces longer than eight bars still use an image of the same size. Bars already written stay on the grid, empty cells are left for the unwritten bars, and the whole image is denoised so that the empty part becomes new notes. The paper’s example starts with four bars, keeps four bars in the image, and writes the next four. Five repetitions produce 24 bars. Notes that already exist are kept as values on the grid.

Chord and texture are conditions given before denoising. The model does not sample these vectors, and it does not decode notes from them. The chords of eight bars are represented by a trained encoder as a 512-dimensional vector. Texture is 256 dimensions for every two bars, and four such blocks are concatenated into 1,024 dimensions. These vectors enter the network as conditions. Denoising produces the roll above. Onset and sustain are then read as notes by a threshold. The notes of this passage are all on that image.

Proffusion uses the same 2×128×128 beat grid [Zhang et al., 2025]. Transformer-Mamba blocks are added to the encoder and decoder of the U-Net. The chord is again given in advance: the eight-bar progression is encoded as a 512-dimensional vector and fed to the network. The generated result is still this roll, and the notes are read from the roll.

When the canvas grows longer but stays in pixel space, Hawley draws the roll as an image and inpaints it with image diffusion [Hawley, 2024]. The Hourglass Diffusion Transformer used there is an image model proposed by Crowson et al. [Crowson et al., 2024]. If the time cell is as fine as 10 ms and the pedal is included, the number of cells becomes too large. Huang et al. first use a variational autoencoder to compress about 1.28 seconds of a three-channel roll, 3×128×128, into a continuous latent of 4×16×16, then concatenate eight segments into 10.24 seconds and denoise them with a DiT [Huang et al., 2024; Peebles and Xie, 2023]. The latent values are real numbers. There is no vocabulary and no codebook index. The object the model accepts is still a continuous object in Class 2. Style is only a class embedding. For lengths beyond 10.24 seconds, Huang et al. use DiffCollage to take overlapping short windows in one denoising pass and merge the scores [Huang et al., 2024]. The decoder returns to the roll, and a threshold turns values into notes. Denoising on pixels and denoising in the latent space both output continuous values.

The two-dimensional table in GETMusic is not a time-by-pitch roll [Lv et al., 2023]. The horizontal axis is time, and one column is one sixteenth note. The vertical axis is the track. Each track takes two rows, one for a pitch symbol and one for a duration symbol. Several notes that sound together on the same track are merged into one pitch symbol, and the duration is the one that occurs most often in the group. The cells hold discrete symbols from a vocabulary. On tracks to be generated, those symbols are replaced by masks, and discrete diffusion predicts the original symbols. Tracks kept as conditions stay as they are. A symbol can be read directly as pitch and duration. Continuous values are not thresholded to 0 or 1. Figure 5 is the six notes of Figure 1 written as this table.

Figure 5. The GETScore of Figure 1. Each column is one sixteenth note [Lv et al., 2023]. (a) Piano. Pitch and duration are written only in the onset column, and duration is a count of sixteenth notes. The other columns are padding. C4 and E4 are merged into one pitch symbol. (b) An unused track; both rows are [EMPTY]. Pitch names are written so the figure can be read. The model stores an index in that track’s vocabulary. Velocity does not enter the table.

Under the division used in this paper, GETMusic remains in Class 2: the model accepts a table aligned in time and track. It touches Class 3 in that the cells are already discrete symbols, and diffusion predicts the original vocabulary symbol rather than a continuous cell value. It does not enter Class 3. A Class-3 code has no pitch name, covers more than one note, and waits for a decoder. The symbols here carry pitch and duration, and they correspond to the notes of one instrument in one time cell.

Every column of a roll occupies a place. The column width uses the two grids of Section 2. On POP909 the cells follow the marked beats, and Polyffusion uses four cells per beat [Min et al., 2023]. On MAESTRO, Huang et al. use 10 ms columns, and Zhang et al. use 32 Hz, about 31 ms per column [Huang et al., 2024; Zhang et al., 2024]. The time shift of Section 3.1 is written only where an event occurs.

Multi-track music in this class is not written into one event sequence, and the instrument is not a field of a token. MuseGAN writes one roll each for bass, drums, guitar, piano, and strings. The five rolls are aligned in time, and the model sees all five at once [Dong et al., 2018]. In GETMusic the track is the vertical axis of one table [Lv et al., 2023]. The onset and sustain channels earlier in this section are two value channels of one roll. They separate the attack from the continuation. They are not two instruments.


## 5. Class 3: sequence learning on coarse codes

A step in the first two classes can still be matched to a note or a cell. Class 1 therefore stretches a piece into a long event sequence, and Class 2 denoises a table that is mostly zeros. Class 3 first compresses a span longer than one note into discrete codes. The downstream model is still a sequence model, and it predicts codebook indices. An index has no pitch name. The notes wait for a decoder, and quantization loses detail.

Figure 6. Lengths of Class-3 index sequences. The counts come from the papers, not from encoding Figure 1. (a) An eight-bar roll becomes 32 indices. (b) One piano roll becomes 1,408 indices, on the same horizontal scale as (a). (c) One bar becomes 8 or 16 residual indices. The unit is the bar, so (c) is not compared in length with (a) and (b).


### 5.2 What span a code covers

All three papers have the downstream model predict discrete indices. They differ in how long the span is before quantization, and in whether an index can be aligned to a bar.

Han et al. quantize an eight-bar bass-and-drum roll into 32 indices from a codebook of 512 entries, predict those indices with an LSTM, and decode them back to a roll [Han et al., 2022]. This is an early placement of generation on discrete codes. The 32 indices are spread over the whole loop. One code covers several time steps of the loop. The unit is not yet the bar, and an index cannot be read as a pitch name.

Zhang et al. quantize a long piano roll into 1,408 indices, generate on those indices with a discrete-diffusion Transformer, and condition on composer style [Zhang et al., 2024]. This discrete diffusion, like SMDIM in Class 1, takes a sequence of symbols as its object. In the forward process of Zhang et al., a symbol may be masked or randomly replaced by another codebook entry. The network predicts the original codebook index. An index cannot be expanded into pitch or duration. It waits for the quantization decoder to return to the roll. One roll is still 1,408 indices.

MuseTok starts from one bar of REMI+ and writes that bar as a stack of indices with residual vector quantization, an RQ-VAE [Huang et al., 2025]. This is still the vector-quantized variational autoencoder setup: an encoder, several codebooks, and a decoder are trained together. The small model uses 8 indices per bar, each from a codebook of 1,024 entries. The large model uses 16 indices per bar, each from a codebook of 2,048 entries. Earlier codebooks follow rhythm and melodic contour more closely. Later codebooks vary more with absolute pitch. A Transformer predicts these indices, and a decoder restores the REMI+ events of that bar. One bar corresponds to this stack. It does not decode an eight-bar roll. The same indices are used both for continuation and for understanding tasks such as chord, texture, and emotion. Reconstruction is lossy.


### 5.3 Models that accept these codes

Han et al. and Zhang et al. do not give the representation a separate name. Both quantize with a vector-quantized variational autoencoder, written VQ-VAE in the table. The encoder maps a span of music to vectors. Each vector is replaced by the nearest entry in a codebook learned during training, and the downstream model predicts that entry’s index. The index has no pitch name. Reconstruction waits for the decoder, and reconstruction is lossy [Han et al., 2022; Zhang et al., 2024].

Byte-pair encoding is described in Section 3.3. After expansion it is still note-level events, so it stays in Class 1. Continuous bar-level latents, such as those in MuseMorphose, are also coarser than one note, but the downstream model does not predict discrete codes [Wu and Yang, 2023], so they do not enter this class. The variational autoencoder of Huang et al. (2024) likewise outputs continuous latents, returns to the roll after denoising, and belongs in Class 2.

A code index in this class has no instrument field, and the music is not split into one sequence per track. Before quantization, Han et al. put bass and drums on one roll: bass uses C1 to B4, and if several pitches sound together only the lowest is kept. Drums are reduced to nine instruments and combined with the bass into 57 rows, over eight bars of 16 cells [Han et al., 2022]. Zhang et al. quantize MAESTRO piano, and the material has only one instrument [Zhang et al., 2024]. MuseTok compresses one bar of events into a stack of codes. The code number itself does not label an instrument [Huang et al., 2025].


## 6. Conclusion

Figures 1 and 3–5 place the first two classes on the same six notes: the notes stay fixed, and the object accepted by the model changes. Figure 2 only delays the onset of E4. The grid is chosen in Section 2. Class 3 is in Figure 6, and the code counts come from the papers. When an event sequence is long, meter is written into the vocabulary and several attributes are gathered into one step. The common learning setup is next-symbol prediction; the same sequence can also use masked diffusion. Preprocessing for note-level events is still file reading and quantization, with no trained encoder. Representing note-level events and rhythm-level events together in one sequence is awkward. Class 2 gives the two-dimensional piano roll to convolution and diffusion and gains polyphony across the grid. The cost is a sparse roll, and passages longer than the fixed canvas have to be generated in segments. Going further, the quantized unit is placed on a bar or on a group of notes. A trained encoder quantizes that span into coarser discrete codes, a sequence model predicts the codes, and reconstruction loses information. Recent models still mostly accept the first two objects: MusicMamba and SMDIM still use REMI or an extension of it, and Proffusion takes a roll. The change is mainly in the backbone, mixing Mamba and attention, and discrete diffusion can also be run on event sequences [Chen et al., 2025; Xu et al., 2026; Zhang et al., 2025].

A general representation that follows levels of musical meaning is still being developed. Waveform, MIDI, and ABC are not yet well covered by one foundation model [Ma et al., 2024]. What deserves more attention is the adaptively learned discrete code in Class 3: the codebook is trained on data, the quantized unit sits near the beat, the bar, or a longer span divided by musical content, and the same indices can be used both to generate and to understand musical content such as chord and texture. MuseTok already compresses one bar of REMI+ into residual indices, and it is one example of this direction [Huang et al., 2025].


## References

The following references are shared by the Chinese text and the English translation. The entries are already in English and are not translated again.

Chen, J., Tang, X., Xie, T., Wang, J., Dong, W., Shi, B. (2025). MusicMamba: A Dual-Feature Modeling Approach for Generating Chinese Traditional Music with Modal Precision. ICASSP. arXiv:2409.02421.

Crowson, K., Baumann, S. A., Birch, A., Abraham, T. M., Kaplan, D. Z., Shippole, E. (2024). Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers. arXiv:2401.11605.

Dong, H.-W., Hsiao, W.-Y., Yang, L.-C., Yang, Y.-H. (2018). MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment. AAAI. arXiv:1709.06298.

Fradet, N., Briot, J.-P., Chhel, F., El Fallah Seghrouchni, A., Gutowski, N. (2021). MidiTok: A Python Package for MIDI File Tokenization. ISMIR Late-Breaking Demo.

Fradet, N., Gutowski, N., Chhel, F., Briot, J.-P. (2023). Byte Pair Encoding for Symbolic Music. EMNLP, pp. 2001–2020. arXiv:2301.11975.

Foscarin, F., McLeod, A., Rigaux, P., Jacquemard, F., Sakai, M. (2020). ASAP: A Dataset of Aligned Scores and Performances for Piano Transcription. ISMIR, pp. 534–541.

Han, S., Ihm, H., Lee, M., Lim, W. (2022). Symbolic Music Loop Generation with Neural Discrete Representations. ISMIR, pp. 403–410. arXiv:2208.05605.

Hawley, S. H. (2024). Pictures Of MIDI: Controlled Music Generation via Graphical Prompts for Image-Based Diffusion Inpainting. arXiv:2407.01499.

Hawthorne, C., Stasyuk, A., Roberts, A., Simon, I., Huang, C.-Z. A., Dieleman, S., Elsen, E., Engel, J., Eck, D. (2019). Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset. ICLR. arXiv:1810.12247.

Hsiao, W.-Y., Liu, J.-Y., Yeh, Y.-C., Yang, Y.-H. (2021). Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs. AAAI. arXiv:2101.02402.

Huang, C.-Z. A., Cooijmans, T., Roberts, A., Courville, A., Eck, D. (2017). Counterpoint by Convolution. ISMIR. arXiv:1903.07227.

Huang, C.-Z. A., Vaswani, A., Uszkoreit, J., Simon, I., Hawthorne, C., Shazeer, N., Dai, A. M., Hoffman, M. D., Dinculescu, M., Eck, D. (2019). Music Transformer. ICLR. arXiv:1809.04281.

Huang, J., Novack, Z., Long, P., Hou, Y., Chen, K., Berg-Kirkpatrick, T., McAuley, J. (2025). MuseTok: Symbolic Music Tokenization for Generation and Semantic Understanding. arXiv:2510.16273.

Huang, Y., Ghatare, A., Liu, Y., Hu, Z., Zhang, Q., Sastry, C. S., Gururani, S., Oore, S., Yue, Y. (2024). Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion. ICML. arXiv:2402.14285.

Huang, Y.-S., Yang, Y.-H. (2020). Pop Music Transformer: Beat-Based Modeling and Generation of Expressive Pop Piano Compositions. ACM MM. arXiv:2002.00212.

Ji, S., Luo, J., Yang, X. (2020). A Comprehensive Survey on Deep Music Generation: Multi-level Representations, Algorithms, Evaluations, and Future Directions. arXiv:2011.06801.

Kong, Q., Li, B., Chen, J., Wang, Y. (2022). GiantMIDI-Piano: A Large-Scale MIDI Dataset for Classical Piano Music. Transactions of the International Society for Music Information Retrieval, 5(1), 87–98.

Le, D.-V.-T., Bigo, L., Keller, M., Herremans, D. (2025). Natural Language Processing Methods for Symbolic Music Generation and Information Retrieval: A Survey. ACM Computing Surveys. arXiv:2402.17467.

Li, S., Ji, S., Wang, Z., Wu, S., Yu, J., Zhang, K. (2025). A Survey on Music Generation from Single-Modal, Cross-Modal and Multi-Modal Perspectives.

Long, P., Novack, Z., Berg-Kirkpatrick, T., McAuley, J. (2024). PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing. arXiv:2409.10831.

Lv, A., Tan, X., Lu, P., Ye, W., Zhang, S., Bian, J., Yan, R. (2023). GETMusic: Generating Music Tracks with a Unified Representation and Diffusion Framework. arXiv:2305.10841.

Ma, Y., Øland, A., Ragni, A., et al. (2024). Foundation Models for Music: A Survey. arXiv:2408.14340.

Min, L., Jiang, J., Xia, G., Zhao, J. (2023). Polyffusion: A Diffusion Model for Polyphonic Score Generation with Internal and External Controls. ISMIR. arXiv:2307.10304.

Oore, S., Simon, I., Dieleman, S., Eck, D., Simonyan, K. (2020). This Time with Feeling: Learning Expressive Musical Performance. Neural Computing and Applications. arXiv:1808.03715.

Peebles, W., Xie, S. (2023). Scalable Diffusion Models with Transformers. ICCV.

Plasser, M., Peter, S., Widmer, G. (2023). Discrete Diffusion Probabilistic Models for Symbolic Music Generation. arXiv:2305.09489.

Raffel, C. (2016). Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching. PhD thesis, Columbia University.

Ren, Y., He, J., Tan, X., Qin, T., Zhao, Z., Liu, T.-Y. (2020). PopMAG: Pop Music Accompaniment Generation. ACM MM, pp. 1198–1206.

Walshaw, C. (2011). The abc Music Standard 2.1. https://abcnotation.com/wiki/abc:standard:v2.1

Wang, Z., Chen, K., Jiang, J., Zhang, Y., Xu, M., Dai, S., Xia, G. (2020). POP909: A Pop-song Dataset for Music Arrangement Generation. ISMIR. arXiv:2008.07142.

Wu, S.-L., Yang, Y.-H. (2023). MuseMorphose: Full-Song and Fine-Grained Piano Music Style Transfer with One Transformer VAE. IEEE/ACM Transactions on Audio, Speech, and Language Processing.

Xu, J., Tang, X., Yang, H., Zhang, H., Yuan, S., Chen, J., Xie, T., Wang, J., Yu, J., Xiang, G. (2026). Efficient Long-Sequence Diffusion Modeling for Symbolic Music Generation. arXiv:2603.00576.

Yang, L.-C., Chou, S.-Y., Yang, Y.-H. (2017). MidiNet: A Convolutional Generative Adversarial Network for Symbolic-Domain Music Generation. ISMIR.

Yeh, Y.-C. (n.d.). Lead Sheet Dataset. https://github.com/wayne391/lead-sheet-dataset

Yuan, R., Lin, H., Wang, Y., Tian, Z., Wu, S., et al. (2024). ChatMusician: Understanding and Generating Music Intrinsically with LLM. Findings of ACL. arXiv:2402.16153.

Zeng, M., Tan, X., Wang, R., Ju, Z., Qin, T., Liu, T.-Y. (2021). MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training. Findings of ACL-IJCNLP. arXiv:2106.05630.

Zhang, H., Tang, J., Rafee, S. R. M., Dixon, S., Fazekas, G. (2022). ATEPP: A Dataset of Automatically Transcribed Expressive Piano Performance. ISMIR, pp. 446–453.

Zhang, J., Fazekas, G., Saitis, C. (2024). Composer Style-specific Symbolic Music Generation Using Vector Quantized Discrete Diffusion Models. IEEE MLSP. arXiv:2310.14044.

Zhang, J., Fazekas, G., Saitis, C. (2025). Mamba-Diffusion Model with Learnable Wavelet for Controllable Symbolic Music Generation. arXiv:2505.03314.
