#!/bin/bash
docker-compose start
luda -v logs tensorflow/tensorflow:1.1.0
