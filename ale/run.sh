RUNLENGTH_ENC=false
# dont make the name as DISPLAY
DISPLAY_SCR=true
FRAMESKIP=8

GAME_CON=fifo_named
#GAME_CON=internal

FOLD=./roms/

GAME=pong.bin
#GAME=breakout.bin
#GAME=enduro.bin
#GAME=qbert.bin
#GAME=seaquest.bin

./ale -game_controller $GAME_CON -player_agent keyboard_agent -run_length_encoding $RUNLENGTH_ENC -display_screen $DISPLAY_SCR -frame_skip $FRAMESKIP ${FOLD}${GAME}


