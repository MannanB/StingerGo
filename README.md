# Stinger Assist

A GrokBot that turns Stinger-Delay predictions into a leave-time decision.

## What it does

Stinger Assist reads your class schedule and mobility needs, checks predicted shuttle delays, and tells you when to leave and which route to take so you still make class. It accounts for elevators, ramps, and longer transfer times. When a delay is likely to make you late, it drafts a short message you can send to your professor.

## Features

- Persistent memory of your schedule, starting point, and mobility constraints
- Scheduled routines that check delays before class and message you only when action is useful
- Decision output: leave now, leave in X minutes, switch routes, or expect to be late
- Accessibility adjustments for students who cannot sprint or use stairs
- Optional professor-late draft (“I’ll be about 5 minutes late”)
- Works with recorded/replayed delay data for demos when live feeds are unavailable

## Demo

Replay a recorded delay scenario → Bot runs its pre-class routine → receives a concrete recommendation with reasoning and an optional late message draft.

## Built with

GrokBot (memory + routines) + Stinger-Delay predictions
