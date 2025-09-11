#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on September 11, 2025, at 00:52
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'squares'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'Gender': ["Male", "Female", "Non-binary", "Prefer not to say"],
    'Age': [18,19,20,21,22,23,24,'26+'],
    'I consent*': False,
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [900, 1600]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Michael\\OneDrive - Georgia Southern University\\4_RESEARCH\\behavior_contrast\\behavior_contrast_main_v1.0.3\\main_csv_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "exp_2_Instructions" ---
    Instructions_phase_1 = visual.TextStim(win=win, name='Instructions_phase_1',
        text='Clicking on the white square sometimes get you points. When points are are earned, click on the other white square that appears and you will earn your points. Please continue to click on the computer screen to earn points for the duration of the experiment. If you have questions, raise your hand and a researcher will come in and help you.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=1.3, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "main_square_exp_2" ---
    # Run 'Begin Experiment' code from code
    score = 0
    phase_time = core.Clock()  # Start a new timer
    random_interval_set = False
    button = visual.ButtonStim(win, 
        text=' ', font='Arvo',
        pos=(0, 0),
        letterHeight=0.05,
        size=(0.5, 0.5), 
        ori=0.0
        ,borderWidth=0.1,
        fillColor=[1.0000, 1.0000, 1.0000], borderColor=[-1.0000, -1.0000, -1.0000],
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-1
    )
    button.buttonClock = core.Clock()
    mouse_logging = event.Mouse(win=win)
    x, y = [None, None]
    mouse_logging.mouseClock = core.Clock()
    textbox_routine_1 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0.38), draggable=False,      letterHeight=0.04,
         size=(0.4, 0.21), borderWidth=0.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.01, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox_routine_1',
         depth=-3, autoLog=False,
    )
    score_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.7, 0.4), draggable=False,      letterHeight=0.1,
         size=(0.15, 0.12), borderWidth=0.0,
         color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='score_textbox',
         depth=-4, autoLog=False,
    )
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(-.5, 0.1), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "points_square_exp_2" ---
    button_2 = visual.ButtonStim(win, 
        text=' ', font='Arvo',
        pos=(0.5, -0.3),
        letterHeight=0.05,
        size=(0.2, 0.2), 
        ori=0.0
        ,borderWidth=0.2,
        fillColor=[1.0000, 1.0000, 1.0000], borderColor=[-1.0000, -1.0000, -1.0000],
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_2',
        depth=0
    )
    button_2.buttonClock = core.Clock()
    textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0.3), draggable=False,      letterHeight=0.05,
         size=(0.4, 0.3), borderWidth=0.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox',
         depth=-1, autoLog=False,
    )
    mouse_logging_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_logging_2.mouseClock = core.Clock()
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=(-.5, 0.1), draggable=False, height=0.02, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "routine_1" ---
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    Instructions_phase = visual.TextStim(win=win, name='Instructions_phase',
        text=' ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=1.3, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "exp_2_Instructions" ---
    # create an object to store info about Routine exp_2_Instructions
    exp_2_Instructions = data.Routine(
        name='exp_2_Instructions',
        components=[Instructions_phase_1, mouse_3],
    )
    exp_2_Instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_3
    mouse_3.x = []
    mouse_3.y = []
    mouse_3.leftButton = []
    mouse_3.midButton = []
    mouse_3.rightButton = []
    mouse_3.time = []
    gotValidClick = False  # until a click is received
    # store start times for exp_2_Instructions
    exp_2_Instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    exp_2_Instructions.tStart = globalClock.getTime(format='float')
    exp_2_Instructions.status = STARTED
    thisExp.addData('exp_2_Instructions.started', exp_2_Instructions.tStart)
    exp_2_Instructions.maxDuration = None
    # keep track of which components have finished
    exp_2_InstructionsComponents = exp_2_Instructions.components
    for thisComponent in exp_2_Instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "exp_2_Instructions" ---
    exp_2_Instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Instructions_phase_1* updates
        
        # if Instructions_phase_1 is starting this frame...
        if Instructions_phase_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Instructions_phase_1.frameNStart = frameN  # exact frame index
            Instructions_phase_1.tStart = t  # local t and not account for scr refresh
            Instructions_phase_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Instructions_phase_1, 'tStartRefresh')  # time at next scr refresh
            # update status
            Instructions_phase_1.status = STARTED
            Instructions_phase_1.setAutoDraw(True)
        
        # if Instructions_phase_1 is active this frame...
        if Instructions_phase_1.status == STARTED:
            # update params
            pass
        # *mouse_3* updates
        
        # if mouse_3 is starting this frame...
        if mouse_3.status == NOT_STARTED and t >= 2-frameTolerance:
            # keep track of start time/frame for later
            mouse_3.frameNStart = frameN  # exact frame index
            mouse_3.tStart = t  # local t and not account for scr refresh
            mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_3.status = STARTED
            prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
        if mouse_3.status == STARTED:  # only update if started and not finished!
            buttons = mouse_3.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_3.getPos()
                    mouse_3.x.append(x)
                    mouse_3.y.append(y)
                    buttons = mouse_3.getPressed()
                    mouse_3.leftButton.append(buttons[0])
                    mouse_3.midButton.append(buttons[1])
                    mouse_3.rightButton.append(buttons[2])
                    mouse_3.time.append(globalClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=exp_2_Instructions,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            exp_2_Instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in exp_2_Instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "exp_2_Instructions" ---
    for thisComponent in exp_2_Instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for exp_2_Instructions
    exp_2_Instructions.tStop = globalClock.getTime(format='float')
    exp_2_Instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('exp_2_Instructions.stopped', exp_2_Instructions.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_3.x', mouse_3.x)
    thisExp.addData('mouse_3.y', mouse_3.y)
    thisExp.addData('mouse_3.leftButton', mouse_3.leftButton)
    thisExp.addData('mouse_3.midButton', mouse_3.midButton)
    thisExp.addData('mouse_3.rightButton', mouse_3.rightButton)
    thisExp.addData('mouse_3.time', mouse_3.time)
    # Run 'End Routine' code from code_3
    globalClock.reset()  # Resets the global timer to 0
    stop_routine = False
    thisExp.nextEntry()
    # the Routine "exp_2_Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials2 = data.TrialHandler2(
        name='trials2',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('bin/.csv/main_loop.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials2)  # add the loop to the experiment
    thisTrials2 = trials2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials2.rgb)
    if thisTrials2 != None:
        for paramName in thisTrials2:
            globals()[paramName] = thisTrials2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials2 in trials2:
        trials2.status = STARTED
        if hasattr(thisTrials2, 'status'):
            thisTrials2.status = STARTED
        currentLoop = trials2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials2.rgb)
        if thisTrials2 != None:
            for paramName in thisTrials2:
                globals()[paramName] = thisTrials2[paramName]
        
        # set up handler to look after randomisation of conditions etc
        Trials_exp_2 = data.TrialHandler2(
            name='Trials_exp_2',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(compnt_file), 
            seed=None, 
        )
        thisExp.addLoop(Trials_exp_2)  # add the loop to the experiment
        thisTrials_exp_2 = Trials_exp_2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_exp_2.rgb)
        if thisTrials_exp_2 != None:
            for paramName in thisTrials_exp_2:
                globals()[paramName] = thisTrials_exp_2[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrials_exp_2 in Trials_exp_2:
            Trials_exp_2.status = STARTED
            if hasattr(thisTrials_exp_2, 'status'):
                thisTrials_exp_2.status = STARTED
            currentLoop = Trials_exp_2
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_exp_2.rgb)
            if thisTrials_exp_2 != None:
                for paramName in thisTrials_exp_2:
                    globals()[paramName] = thisTrials_exp_2[paramName]
            
            # --- Prepare to start Routine "main_square_exp_2" ---
            # create an object to store info about Routine main_square_exp_2
            main_square_exp_2 = data.Routine(
                name='main_square_exp_2',
                components=[button, mouse_logging, textbox_routine_1, score_textbox, text],
            )
            main_square_exp_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            random_interval_set = False
            try:
                min_time = thisTrials_exp_2.data['min_time']
                max_time = thisTrials_exp_2.data['max_time']
                print('Set  min and max time correctly')
            except Exception as e:
                print(f"Error converting min_time or max_time to int: {e}")
                min_time, max_time = None, None
            
            try:
                print('Beginning routine')
            
                # Ensure min_time and max_time are valid before using them
                if min_time is None or max_time is None:
                    raise ValueError("min_time or max_time is None")
            
                random_interval = randint(min_time, max_time)
                random_interval_set = True
                print('Set random interval correctly in the begin routine section!')
            
            except Exception as e:
                random_interval = 15  # Default fallback value
                print('Some general exception occurred!', e)
                print(f'While that was going down, min time was {min_time}, max time was {max_time}')
                print(f'Set random interval to {random_interval}')
                print("This error is nothing to take lightly.  If this happens DURING a trial (and not just between them) it can mean bad stuff.  Otherwise, it's safe to ignore it.")
            
            # Ensure 'background' is defined before using it
            # If 'background' is undefined, this will cause an error
            win.color = background  # (1.0000, 1.0000, 1.0000)
            
            clicked = False
            
            # Uncomment if needed:
            # if instructions_text is None:
            #     instructions_text = ' '
            
            # if phase_time.getTime() > end_time:
            #     continueRoutine = False
            # reset button to account for continued clicks & clear times on/off
            button.reset()
            # setup some python lists for storing info about the mouse_logging
            mouse_logging.x = []
            mouse_logging.y = []
            mouse_logging.leftButton = []
            mouse_logging.midButton = []
            mouse_logging.rightButton = []
            mouse_logging.time = []
            gotValidClick = False  # until a click is received
            textbox_routine_1.reset()
            textbox_routine_1.setText(instructions_text if instructions_text is not None else ' '
            )
            score_textbox.reset()
            score_textbox.setText(score)
            # store start times for main_square_exp_2
            main_square_exp_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            main_square_exp_2.tStart = globalClock.getTime(format='float')
            main_square_exp_2.status = STARTED
            thisExp.addData('main_square_exp_2.started', main_square_exp_2.tStart)
            main_square_exp_2.maxDuration = None
            # skip Routine main_square_exp_2 if its 'Skip if' condition is True
            main_square_exp_2.skipped = continueRoutine and not (stop_routine == True)
            continueRoutine = main_square_exp_2.skipped
            # keep track of which components have finished
            main_square_exp_2Components = main_square_exp_2.components
            for thisComponent in main_square_exp_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "main_square_exp_2" ---
            main_square_exp_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_exp_2, 'status') and thisTrials_exp_2.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code
                # --- Phase timeout check ---
                try:
                    # Ensure end_time is a number before comparison
                    if end_time is not None:
                        # Convert to float just in case it's not already numeric
                        if float(globalClock.getTime()) > float(end_time):
                            print('Phase time expired, stopped routine!')
                            continueRoutine = False
                            stop_routine = True
                    else:
                        print("Warning: end_time is None, skipping phase timeout check.")
                except Exception as e:
                    print(f"Error checking phase timeout: {e}")
                    # Fail-safe: stop routine if timing breaks
                    continueRoutine = False
                    stop_routine = True
                
                
                # --- Random interval section ---
                if not random_interval_set:
                    print('Random interval was not set! Trying to set now!')
                    try:
                        # Safely convert min_time and max_time to integers
                        min_time = int(float(thisTrials_exp_2.data['min_time']))
                        max_time = int(float(thisTrials_exp_2.data['max_time']))
                
                        random_interval = randint(min_time, max_time)
                        random_interval_set = True
                        print('Successfully set random interval in each frame section')
                    except Exception as e:
                        print(f"Failed to set random interval: {e}")
                        print('Throwing in the towel! Forcing random interval to 15 seconds.')
                        random_interval = 15
                
                
                
                # --- Format safe values for display ---
                # --- Trial info safe wrappers ---
                # --- Reset debug vars each frame ---
                trial_num = None
                trial_time = None
                global_time = None
                min_time_val = None
                max_time_val = None
                rand_interval_val = None
                end_time_val = None
                
                interval_reached = False
                phase_expired = False
                can_score = False
                
                cont_routine_flag = None
                stop_routine_flag = None
                clicked_flag = None
                
                # --- Trial info safe wrappers ---
                try:
                    trial_num = thisTrials_exp_2.thisTrialN + 1
                except:
                    pass
                
                try:
                    trial_time = phase_time.getTime()
                except:
                    pass
                
                try:
                    global_time = globalClock.getTime()
                except:
                    pass
                
                try:
                    min_time_val = float(min_time) if min_time is not None else None
                except:
                    pass
                
                try:
                    max_time_val = float(max_time) if max_time is not None else None
                except:
                    pass
                
                try:
                    rand_interval_val = float(random_interval)
                except:
                    pass
                
                try:
                    end_time_val = float(end_time)
                except:
                    pass
                
                # --- Logic checks ---
                try:
                    if trial_time is not None and rand_interval_val is not None:
                        interval_reached = trial_time >= rand_interval_val
                except:
                    pass
                
                try:
                    if end_time_val is not None and global_time is not None:
                        phase_expired = global_time > end_time_val
                except:
                    pass
                
                try:
                    can_score = interval_reached and not phase_expired
                except:
                    pass
                
                # --- Safe flags ---
                try:
                    cont_routine_flag = continueRoutine
                except:
                    pass
                
                try:
                    stop_routine_flag = stop_routine
                except:
                    pass
                
                try:
                    clicked_flag = clicked
                except:
                    pass
                
                # --- Build debug string ---
                # --- Build debug string ---
                debug_status = """
                Component File: {}
                Trial: {}
                Trial Time: {}
                Global Clock: {}
                Score: {}
                
                min_time: {}
                max_time: {}
                Random Interval: {}
                
                Interval Reached? {}
                Phase Expired? {}
                Can Score? {}
                
                continueRoutine: {}
                stop_routine: {}
                Random Interval Set: {}
                Clicked: {}
                """.format(
                    compnt_file if 'compnt_file' in locals() else "NA",
                    trial_num if trial_num is not None else "NA",
                    f"{trial_time:.2f}s" if trial_time is not None else "NA",
                    f"{global_time:.2f}s" if global_time is not None else "NA",
                    score if "score" in locals() else "NA",
                    min_time_val if min_time_val is not None else "NA",
                    max_time_val if max_time_val is not None else "NA",
                    rand_interval_val if rand_interval_val is not None else "NA",
                    interval_reached,
                    phase_expired,
                    can_score,
                    cont_routine_flag,
                    stop_routine_flag,
                    random_interval_set if "random_interval_set" in locals() else "NA",
                    clicked_flag
                )
                
                # *button* updates
                
                # if button is starting this frame...
                if button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button.frameNStart = frameN  # exact frame index
                    button.tStart = t  # local t and not account for scr refresh
                    button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    button.status = STARTED
                    win.callOnFlip(button.buttonClock.reset)
                    button.setAutoDraw(True)
                
                # if button is active this frame...
                if button.status == STARTED:
                    # update params
                    pass
                    # check whether button has been pressed
                    if button.isClicked:
                        if not button.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button.timesOn.append(globalClock.getTime())
                            button.timesOff.append(globalClock.getTime())
                        elif len(button.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button.timesOff[-1] = globalClock.getTime()
                        if not button.wasClicked:
                            # run callback code when button is clicked
                            if t >= random_interval:  # If the routine has been running for 15 seconds
                                continueRoutine = False  # End the routine
                                if points is not None:
                                    try:
                                        if float(points) >= 0:
                                            clicked = True
                                            # score += float(points)  # re-enable if needed
                                    except Exception as e:
                                        print(f"Points value is invalid: {points} ({e})")
                                else:
                                    print("Warning: points is None, skipping points check.")
                # take note of whether button was clicked, so that next frame we know if clicks are new
                button.wasClicked = button.isClicked and button.status == STARTED
                # *mouse_logging* updates
                
                # if mouse_logging is starting this frame...
                if mouse_logging.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_logging.frameNStart = frameN  # exact frame index
                    mouse_logging.tStart = t  # local t and not account for scr refresh
                    mouse_logging.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_logging, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    mouse_logging.status = STARTED
                    prevButtonState = mouse_logging.getPressed()  # if button is down already this ISN'T a new click
                if mouse_logging.status == STARTED:  # only update if started and not finished!
                    buttons = mouse_logging.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                            x, y = mouse_logging.getPos()
                            mouse_logging.x.append(x)
                            mouse_logging.y.append(y)
                            buttons = mouse_logging.getPressed()
                            mouse_logging.leftButton.append(buttons[0])
                            mouse_logging.midButton.append(buttons[1])
                            mouse_logging.rightButton.append(buttons[2])
                            mouse_logging.time.append(globalClock.getTime())
                
                # *textbox_routine_1* updates
                
                # if textbox_routine_1 is starting this frame...
                if textbox_routine_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox_routine_1.frameNStart = frameN  # exact frame index
                    textbox_routine_1.tStart = t  # local t and not account for scr refresh
                    textbox_routine_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox_routine_1, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    textbox_routine_1.status = STARTED
                    textbox_routine_1.setAutoDraw(True)
                
                # if textbox_routine_1 is active this frame...
                if textbox_routine_1.status == STARTED:
                    # update params
                    pass
                
                # *score_textbox* updates
                
                # if score_textbox is starting this frame...
                if score_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    score_textbox.frameNStart = frameN  # exact frame index
                    score_textbox.tStart = t  # local t and not account for scr refresh
                    score_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(score_textbox, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    score_textbox.status = STARTED
                    score_textbox.setAutoDraw(True)
                
                # if score_textbox is active this frame...
                if score_textbox.status == STARTED:
                    # update params
                    pass
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    text.setText(debug_status, log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=main_square_exp_2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    main_square_exp_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in main_square_exp_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "main_square_exp_2" ---
            for thisComponent in main_square_exp_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for main_square_exp_2
            main_square_exp_2.tStop = globalClock.getTime(format='float')
            main_square_exp_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('main_square_exp_2.stopped', main_square_exp_2.tStop)
            Trials_exp_2.addData('button.numClicks', button.numClicks)
            if button.numClicks:
               Trials_exp_2.addData('button.timesOn', button.timesOn)
               Trials_exp_2.addData('button.timesOff', button.timesOff)
            else:
               Trials_exp_2.addData('button.timesOn', "")
               Trials_exp_2.addData('button.timesOff', "")
            # store data for Trials_exp_2 (TrialHandler)
            Trials_exp_2.addData('mouse_logging.x', mouse_logging.x)
            Trials_exp_2.addData('mouse_logging.y', mouse_logging.y)
            Trials_exp_2.addData('mouse_logging.leftButton', mouse_logging.leftButton)
            Trials_exp_2.addData('mouse_logging.midButton', mouse_logging.midButton)
            Trials_exp_2.addData('mouse_logging.rightButton', mouse_logging.rightButton)
            Trials_exp_2.addData('mouse_logging.time', mouse_logging.time)
            # the Routine "main_square_exp_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "points_square_exp_2" ---
            # create an object to store info about Routine points_square_exp_2
            points_square_exp_2 = data.Routine(
                name='points_square_exp_2',
                components=[button_2, textbox, mouse_logging_2, text_2],
            )
            points_square_exp_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # reset button_2 to account for continued clicks & clear times on/off
            button_2.reset()
            textbox.reset()
            textbox.setText(instructions_2_text if instructions_2_text is not None else ' ')
            # setup some python lists for storing info about the mouse_logging_2
            mouse_logging_2.x = []
            mouse_logging_2.y = []
            mouse_logging_2.leftButton = []
            mouse_logging_2.midButton = []
            mouse_logging_2.rightButton = []
            mouse_logging_2.time = []
            gotValidClick = False  # until a click is received
            # store start times for points_square_exp_2
            points_square_exp_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            points_square_exp_2.tStart = globalClock.getTime(format='float')
            points_square_exp_2.status = STARTED
            thisExp.addData('points_square_exp_2.started', points_square_exp_2.tStart)
            points_square_exp_2.maxDuration = 3
            # skip Routine points_square_exp_2 if its 'Skip if' condition is True
            points_square_exp_2.skipped = continueRoutine and not (clicked==False)
            continueRoutine = points_square_exp_2.skipped
            # keep track of which components have finished
            points_square_exp_2Components = points_square_exp_2.components
            for thisComponent in points_square_exp_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "points_square_exp_2" ---
            points_square_exp_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_exp_2, 'status') and thisTrials_exp_2.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > points_square_exp_2.maxDuration-frameTolerance:
                    points_square_exp_2.maxDurationReached = True
                    continueRoutine = False
                # *button_2* updates
                
                # if button_2 is starting this frame...
                if button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    button_2.frameNStart = frameN  # exact frame index
                    button_2.tStart = t  # local t and not account for scr refresh
                    button_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    button_2.status = STARTED
                    win.callOnFlip(button_2.buttonClock.reset)
                    button_2.setAutoDraw(True)
                
                # if button_2 is active this frame...
                if button_2.status == STARTED:
                    # update params
                    pass
                    # check whether button_2 has been pressed
                    if button_2.isClicked:
                        if not button_2.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button_2.timesOn.append(globalClock.getTime())
                            button_2.timesOff.append(globalClock.getTime())
                        elif len(button_2.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button_2.timesOff[-1] = globalClock.getTime()
                        if not button_2.wasClicked:
                            # end routine when button_2 is clicked
                            continueRoutine = False
                        if not button_2.wasClicked:
                            # run callback code when button_2 is clicked
                            score += points
                # take note of whether button_2 was clicked, so that next frame we know if clicks are new
                button_2.wasClicked = button_2.isClicked and button_2.status == STARTED
                
                # *textbox* updates
                
                # if textbox is starting this frame...
                if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox.frameNStart = frameN  # exact frame index
                    textbox.tStart = t  # local t and not account for scr refresh
                    textbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    textbox.status = STARTED
                    textbox.setAutoDraw(True)
                
                # if textbox is active this frame...
                if textbox.status == STARTED:
                    # update params
                    pass
                # *mouse_logging_2* updates
                
                # if mouse_logging_2 is starting this frame...
                if mouse_logging_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_logging_2.frameNStart = frameN  # exact frame index
                    mouse_logging_2.tStart = t  # local t and not account for scr refresh
                    mouse_logging_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_logging_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    mouse_logging_2.status = STARTED
                    prevButtonState = mouse_logging_2.getPressed()  # if button is down already this ISN'T a new click
                if mouse_logging_2.status == STARTED:  # only update if started and not finished!
                    buttons = mouse_logging_2.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                            x, y = mouse_logging_2.getPos()
                            mouse_logging_2.x.append(x)
                            mouse_logging_2.y.append(y)
                            buttons = mouse_logging_2.getPressed()
                            mouse_logging_2.leftButton.append(buttons[0])
                            mouse_logging_2.midButton.append(buttons[1])
                            mouse_logging_2.rightButton.append(buttons[2])
                            mouse_logging_2.time.append(globalClock.getTime())
                # Run 'Each Frame' code from code_2
                if globalClock.getTime() > end_time:#  if t > end_time:
                    print('Phase time expired, stopped routine!')
                #    Trials_exp_2.finished = True
                    continueRoutine = False
                    stop_routine = True
                
                # *text_2* updates
                
                # if text_2 is starting this frame...
                if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_2.frameNStart = frameN  # exact frame index
                    text_2.tStart = t  # local t and not account for scr refresh
                    text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.started')
                    # update status
                    text_2.status = STARTED
                    text_2.setAutoDraw(True)
                
                # if text_2 is active this frame...
                if text_2.status == STARTED:
                    # update params
                    text_2.setText(debug_status, log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=points_square_exp_2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    points_square_exp_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in points_square_exp_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "points_square_exp_2" ---
            for thisComponent in points_square_exp_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for points_square_exp_2
            points_square_exp_2.tStop = globalClock.getTime(format='float')
            points_square_exp_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('points_square_exp_2.stopped', points_square_exp_2.tStop)
            Trials_exp_2.addData('button_2.numClicks', button_2.numClicks)
            if button_2.numClicks:
               Trials_exp_2.addData('button_2.timesOn', button_2.timesOn)
               Trials_exp_2.addData('button_2.timesOff', button_2.timesOff)
            else:
               Trials_exp_2.addData('button_2.timesOn', "")
               Trials_exp_2.addData('button_2.timesOff', "")
            # store data for Trials_exp_2 (TrialHandler)
            Trials_exp_2.addData('mouse_logging_2.x', mouse_logging_2.x)
            Trials_exp_2.addData('mouse_logging_2.y', mouse_logging_2.y)
            Trials_exp_2.addData('mouse_logging_2.leftButton', mouse_logging_2.leftButton)
            Trials_exp_2.addData('mouse_logging_2.midButton', mouse_logging_2.midButton)
            Trials_exp_2.addData('mouse_logging_2.rightButton', mouse_logging_2.rightButton)
            Trials_exp_2.addData('mouse_logging_2.time', mouse_logging_2.time)
            # the Routine "points_square_exp_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTrials_exp_2 as finished
            if hasattr(thisTrials_exp_2, 'status'):
                thisTrials_exp_2.status = FINISHED
            # if awaiting a pause, pause now
            if Trials_exp_2.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                Trials_exp_2.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'Trials_exp_2'
        Trials_exp_2.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if Trials_exp_2.trialList in ([], [None], None):
            params = []
        else:
            params = Trials_exp_2.trialList[0].keys()
        # save data for this loop
        Trials_exp_2.saveAsText(filename + '_Trials_exp_2.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "routine_1" ---
        # create an object to store info about Routine routine_1
        routine_1 = data.Routine(
            name='routine_1',
            components=[mouse, Instructions_phase],
        )
        routine_1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_4
        #win.color =  (0,0,0) #black #background # (1.0000, 1.0000, 1.0000)
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        gotValidClick = False  # until a click is received
        # store start times for routine_1
        routine_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        routine_1.tStart = globalClock.getTime(format='float')
        routine_1.status = STARTED
        thisExp.addData('routine_1.started', routine_1.tStart)
        routine_1.maxDuration = 3
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        routine_1Components = routine_1.components
        for thisComponent in routine_1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "routine_1" ---
        routine_1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrials2, 'status') and thisTrials2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > routine_1.maxDuration-frameTolerance:
                routine_1.maxDurationReached = True
                continueRoutine = False
            # Run 'Each Frame' code from code_4
            #win.color =  (0,0,0) #black #background # (1.0000, 1.0000, 1.0000)
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 2-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse.status = STARTED
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse.getPos()
                        mouse.x.append(x)
                        mouse.y.append(y)
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(globalClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *Instructions_phase* updates
            
            # if Instructions_phase is starting this frame...
            if Instructions_phase.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Instructions_phase.frameNStart = frameN  # exact frame index
                Instructions_phase.tStart = t  # local t and not account for scr refresh
                Instructions_phase.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Instructions_phase, 'tStartRefresh')  # time at next scr refresh
                # update status
                Instructions_phase.status = STARTED
                Instructions_phase.setAutoDraw(True)
            
            # if Instructions_phase is active this frame...
            if Instructions_phase.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=routine_1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routine_1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in routine_1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "routine_1" ---
        for thisComponent in routine_1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for routine_1
        routine_1.tStop = globalClock.getTime(format='float')
        routine_1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('routine_1.stopped', routine_1.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from code_4
        globalClock.reset()  # Resets the global timer to 0
        stop_routine = False
        # store data for trials2 (TrialHandler)
        trials2.addData('mouse.x', mouse.x)
        trials2.addData('mouse.y', mouse.y)
        trials2.addData('mouse.leftButton', mouse.leftButton)
        trials2.addData('mouse.midButton', mouse.midButton)
        trials2.addData('mouse.rightButton', mouse.rightButton)
        trials2.addData('mouse.time', mouse.time)
        # the Routine "routine_1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrials2 as finished
        if hasattr(thisTrials2, 'status'):
            thisTrials2.status = FINISHED
        # if awaiting a pause, pause now
        if trials2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials2'
    trials2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials2.trialList in ([], [None], None):
        params = []
    else:
        params = trials2.trialList[0].keys()
    # save data for this loop
    trials2.saveAsText(filename + '_trials2.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
