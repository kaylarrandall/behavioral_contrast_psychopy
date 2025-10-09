#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on October 09, 2025, at 16:23
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

# Run 'Before Experiment' code from code_2
component_clicks = 0
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'behavioral contrast'  # from the Builder filename that created this script
expVersion = 'v1.1.0'
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'Gender': ["Male", "Female", "Non-binary", "Prefer not to say"],
    'Age': [18,19,20,21,22,23,24,'26+'],
    'I consent*': False,
    'experiment_selection': ["bin\.csv_long\main_loop_long.csv","bin\.csv_short\main_loop_short.csv"],
    'Debug?': False,
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
_winSize = [1920, 1080]
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
        originPath='C:\\Users\\micha\\OneDrive - Georgia Southern University\\behavioral_contrast_psychopy\\main_v1.1.1_lastrun.py',
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
            winType='pyglet', allowGUI=True, allowStencil=False,
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
    
    # --- Initialize components for Routine "exp_Instructions" ---
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
    
    # --- Initialize components for Routine "main_square_screen" ---
    logging_mouse = event.Mouse(win=win)
    x, y = [None, None]
    logging_mouse.mouseClock = core.Clock()
    click_square = visual.Rect(
        win=win, name='click_square',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='black', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    mouse_score = event.Mouse(win=win)
    x, y = [None, None]
    mouse_score.mouseClock = core.Clock()
    debug_text = visual.TextStim(win=win, name='debug_text',
        text='',
        font='Arial',
        pos=(-.5, 0.1), draggable=False, height=0.06, wrapWidth=1.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    # Run 'Begin Experiment' code from code
    score = 0
    component_clicks = 0
    mouseDown = False
    score_main_screen = visual.TextStim(win=win, name='score_main_screen',
        text='',
        font='Arial',
        pos=(-0.8, 0.44), draggable=False, height=0.15, wrapWidth=1.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "points_square_screen" ---
    mouse_logging_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_logging_2.mouseClock = core.Clock()
    button = visual.ButtonStim(win, 
        text=None, font='Arvo',
        pos=(0.5, -0.4),
        letterHeight=0.05,
        size=(0.15, 0.15), 
        ori=0.0
        ,borderWidth=2.0,
        fillColor='white', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-1
    )
    button.buttonClock = core.Clock()
    score_points_screen = visual.TextStim(win=win, name='score_points_screen',
        text='',
        font='Arial',
        pos=(-0.8, 0.44), draggable=False, height=0.15, wrapWidth=1.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "blackout_screen" ---
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
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
    
    # --- Prepare to start Routine "exp_Instructions" ---
    # create an object to store info about Routine exp_Instructions
    exp_Instructions = data.Routine(
        name='exp_Instructions',
        components=[Instructions_phase_1, mouse_3],
    )
    exp_Instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_3
    gotValidClick = False  # until a click is received
    # store start times for exp_Instructions
    exp_Instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    exp_Instructions.tStart = globalClock.getTime(format='float')
    exp_Instructions.status = STARTED
    exp_Instructions.maxDuration = None
    # keep track of which components have finished
    exp_InstructionsComponents = exp_Instructions.components
    for thisComponent in exp_Instructions.components:
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
    
    # --- Run Routine "exp_Instructions" ---
    exp_Instructions.forceEnded = routineForceEnded = not continueRoutine
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
                currentRoutine=exp_Instructions,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            exp_Instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in exp_Instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "exp_Instructions" ---
    for thisComponent in exp_Instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for exp_Instructions
    exp_Instructions.tStop = globalClock.getTime(format='float')
    exp_Instructions.tStopRefresh = tThisFlipGlobal
    # store data for thisExp (ExperimentHandler)
    # Run 'End Routine' code from code_2
    globalClock.reset()
    thisExp.nextEntry()
    # the Routine "exp_Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    outer_trials = data.TrialHandler2(
        name='outer_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(expInfo['experiment_selection']), 
        seed=None, 
    )
    thisExp.addLoop(outer_trials)  # add the loop to the experiment
    thisOuter_trial = outer_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisOuter_trial.rgb)
    if thisOuter_trial != None:
        for paramName in thisOuter_trial:
            globals()[paramName] = thisOuter_trial[paramName]
    
    for thisOuter_trial in outer_trials:
        outer_trials.status = STARTED
        if hasattr(thisOuter_trial, 'status'):
            thisOuter_trial.status = STARTED
        currentLoop = outer_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisOuter_trial.rgb)
        if thisOuter_trial != None:
            for paramName in thisOuter_trial:
                globals()[paramName] = thisOuter_trial[paramName]
        
        # set up handler to look after randomisation of conditions etc
        inner_trials = data.TrialHandler2(
            name='inner_trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(compnt_file), 
            seed=None, 
        )
        thisExp.addLoop(inner_trials)  # add the loop to the experiment
        thisInner_trial = inner_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisInner_trial.rgb)
        if thisInner_trial != None:
            for paramName in thisInner_trial:
                globals()[paramName] = thisInner_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisInner_trial in inner_trials:
            inner_trials.status = STARTED
            if hasattr(thisInner_trial, 'status'):
                thisInner_trial.status = STARTED
            currentLoop = inner_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisInner_trial.rgb)
            if thisInner_trial != None:
                for paramName in thisInner_trial:
                    globals()[paramName] = thisInner_trial[paramName]
            
            # --- Prepare to start Routine "main_square_screen" ---
            # create an object to store info about Routine main_square_screen
            main_square_screen = data.Routine(
                name='main_square_screen',
                components=[logging_mouse, click_square, mouse_score, debug_text, score_main_screen],
            )
            main_square_screen.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # setup some python lists for storing info about the logging_mouse
            logging_mouse.x = []
            logging_mouse.y = []
            logging_mouse.leftButton = []
            logging_mouse.midButton = []
            logging_mouse.rightButton = []
            logging_mouse.time = []
            gotValidClick = False  # until a click is received
            logging_mouse.mouseClock.reset()
            # setup some python lists for storing info about the mouse_score
            mouse_score.x = []
            mouse_score.y = []
            mouse_score.leftButton = []
            mouse_score.midButton = []
            mouse_score.rightButton = []
            mouse_score.time = []
            mouse_score.clicked_name = []
            gotValidClick = False  # until a click is received
            mouse_score.mouseClock.reset()
            # Run 'Begin Routine' code from code
            #thisExp.addData("score", score)
            
            print('Set component clicks to 0 in begin routine section')
            component_clicks = 0
            mouseDown = True
            score_main_screen.setText(score)
            # store start times for main_square_screen
            main_square_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            main_square_screen.tStart = globalClock.getTime(format='float')
            main_square_screen.status = STARTED
            thisExp.addData('main_square_screen.started', main_square_screen.tStart)
            main_square_screen.maxDuration = None
            # skip Routine main_square_screen if its 'Skip if' condition is True
            main_square_screen.skipped = continueRoutine and not (globalClock.getTime() > end_time)
            continueRoutine = main_square_screen.skipped
            win.color = background
            win.colorSpace = 'rgb'
            win.backgroundImage = ''
            win.backgroundFit = 'none'
            # keep track of which components have finished
            main_square_screenComponents = main_square_screen.components
            for thisComponent in main_square_screen.components:
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
            
            # --- Run Routine "main_square_screen" ---
            main_square_screen.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInner_trial, 'status') and thisInner_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # *logging_mouse* updates
                
                # if logging_mouse is starting this frame...
                if logging_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    logging_mouse.frameNStart = frameN  # exact frame index
                    logging_mouse.tStart = t  # local t and not account for scr refresh
                    logging_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(logging_mouse, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    logging_mouse.status = STARTED
                    prevButtonState = logging_mouse.getPressed()  # if button is down already this ISN'T a new click
                
                # if logging_mouse is stopping this frame...
                if logging_mouse.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > logging_mouse.tStartRefresh + scoring_interval-frameTolerance:
                        # keep track of stop time/frame for later
                        logging_mouse.tStop = t  # not accounting for scr refresh
                        logging_mouse.tStopRefresh = tThisFlipGlobal  # on global time
                        logging_mouse.frameNStop = frameN  # exact frame index
                        # update status
                        logging_mouse.status = FINISHED
                if logging_mouse.status == STARTED:  # only update if started and not finished!
                    buttons = logging_mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            pass
                            x, y = logging_mouse.getPos()
                            logging_mouse.x.append(x)
                            logging_mouse.y.append(y)
                            buttons = logging_mouse.getPressed()
                            logging_mouse.leftButton.append(buttons[0])
                            logging_mouse.midButton.append(buttons[1])
                            logging_mouse.rightButton.append(buttons[2])
                            logging_mouse.time.append(logging_mouse.mouseClock.getTime())
                
                # *click_square* updates
                
                # if click_square is starting this frame...
                if click_square.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    click_square.frameNStart = frameN  # exact frame index
                    click_square.tStart = t  # local t and not account for scr refresh
                    click_square.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(click_square, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'click_square.started')
                    # update status
                    click_square.status = STARTED
                    click_square.setAutoDraw(True)
                
                # if click_square is active this frame...
                if click_square.status == STARTED:
                    # update params
                    pass
                # *mouse_score* updates
                
                # if mouse_score is starting this frame...
                if mouse_score.status == NOT_STARTED and t >= scoring_interval-frameTolerance:
                    # keep track of start time/frame for later
                    mouse_score.frameNStart = frameN  # exact frame index
                    mouse_score.tStart = t  # local t and not account for scr refresh
                    mouse_score.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse_score, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    mouse_score.status = STARTED
                    prevButtonState = mouse_score.getPressed()  # if button is down already this ISN'T a new click
                if mouse_score.status == STARTED:  # only update if started and not finished!
                    buttons = mouse_score.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames(click_square, namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse_score):
                                    gotValidClick = True
                                    mouse_score.clicked_name.append(obj.name)
                            if not gotValidClick:
                                mouse_score.clicked_name.append(None)
                            x, y = mouse_score.getPos()
                            mouse_score.x.append(x)
                            mouse_score.y.append(y)
                            buttons = mouse_score.getPressed()
                            mouse_score.leftButton.append(buttons[0])
                            mouse_score.midButton.append(buttons[1])
                            mouse_score.rightButton.append(buttons[2])
                            mouse_score.time.append(mouse_score.mouseClock.getTime())
                            if gotValidClick:
                                continueRoutine = False  # end routine on response
                
                # *debug_text* updates
                
                # if debug_text is starting this frame...
                if debug_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    debug_text.frameNStart = frameN  # exact frame index
                    debug_text.tStart = t  # local t and not account for scr refresh
                    debug_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(debug_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    debug_text.status = STARTED
                    debug_text.setAutoDraw(True)
                
                # if debug_text is active this frame...
                if debug_text.status == STARTED:
                    # update params
                    debug_text.setText((
                        f"component_clicks:{component_clicks}\n"
                        f"score:{score}\n"
                        f"points awarded next: {points}\n"
                        f"trial clock {int(t)}\n"
                        f"score interval:{scoring_interval}\n"
                        f"Global Clock: {int(globalClock.getTime())}\n"
                        f"end time: {end_time}\n\n"
                        f"Core Clock:{int(core.getTime())}\n"
                        f"points: {points}\n"
                        f"background: {background}\n"
                        if expInfo['Debug?']
                        else ''
                    )
                    , log=False)
                # Run 'Each Frame' code from code
                if globalClock.getTime() > end_time:
                    print(f'inner_trials.nRemaining{inner_trials.nRemaining}')
                    print(f'Skipping{inner_trials.nRemaining} trials')
                    print(f'Time: global:{globalClock.getTime()} end time {end_time}')
                    print(f'background{background}')
                    skip = int(inner_trials.nRemaining)
                    inner_trials.skipTrials(skip)
                    continueRoutine = False
                    
                #if logging_mouse.getPressed()[0] == 1 and not mouseDown:
                #    print('clicked')
                #    component_clicks += 1
                #    print(component_clicks)
                #    
                
                
                # Inside your loop:
                # Inside your loop:
                left_pressed = (
                    logging_mouse.getPressed()[0] == 1
                    or mouse_score.getPressed()[0] == 1
                )
                
                if left_pressed and not mouseDown:
                #    print('clicked')
                    component_clicks += 1
                    print(f'component_clicks:{component_clicks}')
                    mouseDown = True  # now it’s pressed
                
                # Reset when mouse is released
                if (
                    logging_mouse.getPressed()[0] == 0
                    and mouse_score.getPressed()[0] == 0
                ):
                    mouseDown = False
                
                
                # *score_main_screen* updates
                
                # if score_main_screen is starting this frame...
                if score_main_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    score_main_screen.frameNStart = frameN  # exact frame index
                    score_main_screen.tStart = t  # local t and not account for scr refresh
                    score_main_screen.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(score_main_screen, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    score_main_screen.status = STARTED
                    score_main_screen.setAutoDraw(True)
                
                # if score_main_screen is active this frame...
                if score_main_screen.status == STARTED:
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
                        currentRoutine=main_square_screen,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    main_square_screen.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in main_square_screen.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "main_square_screen" ---
            for thisComponent in main_square_screen.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for main_square_screen
            main_square_screen.tStop = globalClock.getTime(format='float')
            main_square_screen.tStopRefresh = tThisFlipGlobal
            thisExp.addData('main_square_screen.stopped', main_square_screen.tStop)
            setupWindow(expInfo=expInfo, win=win)
            # store data for inner_trials (TrialHandler)
            inner_trials.addData('logging_mouse.x', logging_mouse.x)
            inner_trials.addData('logging_mouse.y', logging_mouse.y)
            inner_trials.addData('logging_mouse.leftButton', logging_mouse.leftButton)
            inner_trials.addData('logging_mouse.midButton', logging_mouse.midButton)
            inner_trials.addData('logging_mouse.rightButton', logging_mouse.rightButton)
            inner_trials.addData('logging_mouse.time', logging_mouse.time)
            # store data for inner_trials (TrialHandler)
            inner_trials.addData('mouse_score.x', mouse_score.x)
            inner_trials.addData('mouse_score.y', mouse_score.y)
            inner_trials.addData('mouse_score.leftButton', mouse_score.leftButton)
            inner_trials.addData('mouse_score.midButton', mouse_score.midButton)
            inner_trials.addData('mouse_score.rightButton', mouse_score.rightButton)
            inner_trials.addData('mouse_score.time', mouse_score.time)
            inner_trials.addData('mouse_score.clicked_name', mouse_score.clicked_name)
            # Run 'End Routine' code from code
            
            # --- End Routine ---
            thisExp.addData('component_clicks', component_clicks)
            thisExp.addData('score', score)
            component_clicks = 0
            
            #if (
            #    logging_mouse.getPressed()[0] == 0
            #    and mouse_score.getPressed()[0] == 0
            #):
            #    mouseDown = False
            # the Routine "main_square_screen" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "points_square_screen" ---
            # create an object to store info about Routine points_square_screen
            points_square_screen = data.Routine(
                name='points_square_screen',
                components=[mouse_logging_2, button, score_points_screen],
            )
            points_square_screen.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # setup some python lists for storing info about the mouse_logging_2
            mouse_logging_2.x = []
            mouse_logging_2.y = []
            mouse_logging_2.leftButton = []
            mouse_logging_2.midButton = []
            mouse_logging_2.rightButton = []
            mouse_logging_2.time = []
            mouse_logging_2.clicked_name = []
            gotValidClick = False  # until a click is received
            mouse_logging_2.mouseClock.reset()
            # reset button to account for continued clicks & clear times on/off
            button.reset()
            score_points_screen.setText(score)
            # store start times for points_square_screen
            points_square_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            points_square_screen.tStart = globalClock.getTime(format='float')
            points_square_screen.status = STARTED
            thisExp.addData('points_square_screen.started', points_square_screen.tStart)
            points_square_screen.maxDuration = 3
            # skip Routine points_square_screen if its 'Skip if' condition is True
            points_square_screen.skipped = continueRoutine and not (globalClock.getTime() > end_time)
            continueRoutine = points_square_screen.skipped
            win.color = background
            win.colorSpace = 'rgb'
            win.backgroundImage = ''
            win.backgroundFit = 'none'
            # keep track of which components have finished
            points_square_screenComponents = points_square_screen.components
            for thisComponent in points_square_screen.components:
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
            
            # --- Run Routine "points_square_screen" ---
            points_square_screen.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInner_trial, 'status') and thisInner_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > points_square_screen.maxDuration-frameTolerance:
                    points_square_screen.maxDurationReached = True
                    continueRoutine = False
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
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames(button, namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse_logging_2):
                                    gotValidClick = True
                                    mouse_logging_2.clicked_name.append(obj.name)
                            if not gotValidClick:
                                mouse_logging_2.clicked_name.append(None)
                            x, y = mouse_logging_2.getPos()
                            mouse_logging_2.x.append(x)
                            mouse_logging_2.y.append(y)
                            buttons = mouse_logging_2.getPressed()
                            mouse_logging_2.leftButton.append(buttons[0])
                            mouse_logging_2.midButton.append(buttons[1])
                            mouse_logging_2.rightButton.append(buttons[2])
                            mouse_logging_2.time.append(mouse_logging_2.mouseClock.getTime())
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
                            button.timesOn.append(button.buttonClock.getTime())
                            button.timesOff.append(button.buttonClock.getTime())
                        elif len(button.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button.timesOff[-1] = button.buttonClock.getTime()
                        if not button.wasClicked:
                            # end routine when button is clicked
                            continueRoutine = False
                        if not button.wasClicked:
                            # run callback code when button is clicked
                            score += points
                # take note of whether button was clicked, so that next frame we know if clicks are new
                button.wasClicked = button.isClicked and button.status == STARTED
                
                # *score_points_screen* updates
                
                # if score_points_screen is starting this frame...
                if score_points_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    score_points_screen.frameNStart = frameN  # exact frame index
                    score_points_screen.tStart = t  # local t and not account for scr refresh
                    score_points_screen.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(score_points_screen, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    score_points_screen.status = STARTED
                    score_points_screen.setAutoDraw(True)
                
                # if score_points_screen is active this frame...
                if score_points_screen.status == STARTED:
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
                        currentRoutine=points_square_screen,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    points_square_screen.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in points_square_screen.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "points_square_screen" ---
            for thisComponent in points_square_screen.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for points_square_screen
            points_square_screen.tStop = globalClock.getTime(format='float')
            points_square_screen.tStopRefresh = tThisFlipGlobal
            thisExp.addData('points_square_screen.stopped', points_square_screen.tStop)
            setupWindow(expInfo=expInfo, win=win)
            # store data for inner_trials (TrialHandler)
            inner_trials.addData('mouse_logging_2.x', mouse_logging_2.x)
            inner_trials.addData('mouse_logging_2.y', mouse_logging_2.y)
            inner_trials.addData('mouse_logging_2.leftButton', mouse_logging_2.leftButton)
            inner_trials.addData('mouse_logging_2.midButton', mouse_logging_2.midButton)
            inner_trials.addData('mouse_logging_2.rightButton', mouse_logging_2.rightButton)
            inner_trials.addData('mouse_logging_2.time', mouse_logging_2.time)
            inner_trials.addData('mouse_logging_2.clicked_name', mouse_logging_2.clicked_name)
            # Run 'End Routine' code from code_3
            component_clicks = 0
            # the Routine "points_square_screen" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisInner_trial as finished
            if hasattr(thisInner_trial, 'status'):
                thisInner_trial.status = FINISHED
            # if awaiting a pause, pause now
            if inner_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                inner_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'inner_trials'
        inner_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "blackout_screen" ---
        # create an object to store info about Routine blackout_screen
        blackout_screen = data.Routine(
            name='blackout_screen',
            components=[mouse],
        )
        blackout_screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the mouse
        gotValidClick = False  # until a click is received
        mouse.mouseClock.reset()
        # store start times for blackout_screen
        blackout_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blackout_screen.tStart = globalClock.getTime(format='float')
        blackout_screen.status = STARTED
        blackout_screen.maxDuration = 3
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        blackout_screenComponents = blackout_screen.components
        for thisComponent in blackout_screen.components:
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
        
        # --- Run Routine "blackout_screen" ---
        blackout_screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisOuter_trial, 'status') and thisOuter_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > blackout_screen.maxDuration-frameTolerance:
                blackout_screen.maxDurationReached = True
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
                prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
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
                    currentRoutine=blackout_screen,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blackout_screen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blackout_screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blackout_screen" ---
        for thisComponent in blackout_screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blackout_screen
        blackout_screen.tStop = globalClock.getTime(format='float')
        blackout_screen.tStopRefresh = tThisFlipGlobal
        setupWindow(expInfo=expInfo, win=win)
        # Run 'End Routine' code from code_4
        globalClock.reset()  # Resets the global timer to 0
        # stop_routine = False
        component_clicks = 0
        # store data for outer_trials (TrialHandler)
        # the Routine "blackout_screen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisOuter_trial as finished
        if hasattr(thisOuter_trial, 'status'):
            thisOuter_trial.status = FINISHED
        # if awaiting a pause, pause now
        if outer_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            outer_trials.status = STARTED
    # completed 1.0 repeats of 'outer_trials'
    outer_trials.status = FINISHED
    
    
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
