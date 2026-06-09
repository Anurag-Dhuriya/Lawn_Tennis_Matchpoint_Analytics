import cv2
import numpy as np
import constants

COURT_W = 520
COURT_H = int(COURT_W * (constants.HALF_COURT_LINE_HEIGHT * 2) / constants.DOUBLE_LINE_WIDTH)
PAD = 50


def _mini_bounds(mini_court):
    pts = mini_court.get_court_drawing_keypoints()
    left = pts[0]
    right = pts[2]
    top = pts[1]
    bottom = pts[5]
    return left, top, right, bottom


def _to_panel_point(point, bounds):
    left, top, right, bottom = bounds
    x, y = point
    px = PAD + int((x - left) / (right - left) * COURT_W)
    py = PAD + int((y - top) / (bottom - top) * COURT_H)
    return px, py


def _draw_court(canvas):
    x0, y0 = PAD, PAD
    x1, y1 = PAD + COURT_W, PAD + COURT_H

    singles_gap = int(COURT_W * constants.DOUBLE_ALLY_DIFFERENCE / constants.DOUBLE_LINE_WIDTH)
    service_y_gap = int(COURT_H * constants.NO_MANS_LAND_HEIGHT / (constants.HALF_COURT_LINE_HEIGHT * 2))
    center_x = (x0 + x1) // 2
    net_y = (y0 + y1) // 2

    white = (245, 245, 245)

    cv2.rectangle(canvas, (x0, y0), (x1, y1), white, 3)
    cv2.line(canvas, (x0 + singles_gap, y0), (x0 + singles_gap, y1), white, 2)
    cv2.line(canvas, (x1 - singles_gap, y0), (x1 - singles_gap, y1), white, 2)
    cv2.line(canvas, (x0, net_y), (x1, net_y), (35, 90, 210), 4)

    top_service_y = y0 + service_y_gap
    bottom_service_y = y1 - service_y_gap

    cv2.line(canvas, (x0 + singles_gap, top_service_y), (x1 - singles_gap, top_service_y), white, 2)
    cv2.line(canvas, (x0 + singles_gap, bottom_service_y), (x1 - singles_gap, bottom_service_y), white, 2)
    cv2.line(canvas, (center_x, top_service_y), (center_x, bottom_service_y), white, 2)


def _make_heatmap(points, title):
    panel_w = COURT_W + PAD * 2
    panel_h = COURT_H + PAD * 2 + 80

    base = np.full((panel_h, panel_w, 3), (72, 112, 145), dtype=np.uint8)
    court_layer = base.copy()
    _draw_court(court_layer)

    density = np.zeros((panel_h, panel_w), dtype=np.float32)

    for x, y in points:
        if PAD <= x <= PAD + COURT_W and PAD <= y <= PAD + COURT_H:
            cv2.circle(density, (int(x), int(y)), 10, 1, -1)

    density = cv2.GaussianBlur(density, (0, 0), 28)

    if density.max() > 0:
        norm = np.uint8(255 * density / density.max())
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mask = norm > 8
        court_layer[mask] = cv2.addWeighted(court_layer, 0.45, colored, 0.65, 0)[mask]

    _draw_court(court_layer)

    cv2.putText(court_layer, title, (PAD, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return court_layer


def _draw_legend(canvas):
    x, y, w, h = 60, canvas.shape[0] - 55, 300, 22
    gradient = np.linspace(0, 255, w).astype(np.uint8)
    gradient = np.tile(gradient, (h, 1))
    legend = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    canvas[y:y + h, x:x + w] = legend

    cv2.putText(canvas, "Low", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "High", (x + w - 45, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "Heatmap intensity", (x + w + 20, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def save_heatmap_dashboard(
    mini_court,
    player_mini_court_detections,
    ball_mini_court_detections,
    output_path,
    selected_player_id=1
):
    bounds = _mini_bounds(mini_court)

    player_points = []
    for frame_positions in player_mini_court_detections:
        if selected_player_id in frame_positions:
            player_points.append(_to_panel_point(frame_positions[selected_player_id], bounds))

    ball_points = []
    for frame_positions in ball_mini_court_detections:
        if 1 in frame_positions:
            ball_points.append(_to_panel_point(frame_positions[1], bounds))

    player_panel = _make_heatmap(player_points, f"Player {selected_player_id} Movement Heatmap")
    ball_panel = _make_heatmap(ball_points, "Ball Placement Heatmap")

    gap = 35
    dashboard_h = max(player_panel.shape[0], ball_panel.shape[0]) + 90
    dashboard_w = player_panel.shape[1] + ball_panel.shape[1] + gap

    dashboard = np.full((dashboard_h, dashboard_w, 3), (24, 28, 34), dtype=np.uint8)
    dashboard[20:20 + player_panel.shape[0], 0:player_panel.shape[1]] = player_panel
    dashboard[20:20 + ball_panel.shape[0], player_panel.shape[1] + gap:] = ball_panel

    _draw_legend(dashboard)
    cv2.imwrite(output_path, dashboard)