RUNLENGTH_ENC=false
# Toggle display output
DISPLAY_SCR=false
FRAMESKIP=4

# Playing through pipe
GAME_CON=fifo_named
# Playing by internal agent (keyboard)
#GAME_CON=internal

FOLD=./roms/

# Games
#GAME=pong.bin
GAME=breakout.bin
#GAME=seaquest.bin
#GAME=space_invaders.bin
#GAME=beam_rider.bin

# make pipes
mkfifo ale_fifo_in
mkfifo ale_fifo_out
# make folder
mkdir store

./ale -game_controller $GAME_CON -player_agent keyboard_agent -run_length_encoding $RUNLENGTH_ENC -display_screen $DISPLAY_SCR -frame_skip $FRAMESKIP ${FOLD}${GAME} &

python Ale_client.py $GAME
