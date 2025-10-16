close all
clear
clc

file = load("C:\Users\brend\Documents\GitHub\4YPMotionCapture\trajectories\trajectory_H0139_S1000_V0451_20251016_181759.mat");
traj = file.traj;
scale = 1;
traj = traj*scale;

figure
plot3(traj(:,1), traj(:,3),  traj(:,2), '.-', 'Color',"#c49102", 'LineWidth', 1.5);
hold on
xlabel('X (mm)'); ylabel('Z (mm)'); zlabel('Y (mm)');
grid on; axis equal; hold off
title('3D Trajectories');  view(20,30);hold off;