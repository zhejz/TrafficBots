# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from gym.wrappers.monitoring.video_recorder import ImageEncoder

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_VIOLET = (170, 0, 255)

COLOR_BUTTER_0 = (252, 233, 79)
COLOR_BUTTER_1 = (237, 212, 0)
COLOR_BUTTER_2 = (196, 160, 0)
COLOR_ORANGE_0 = (252, 175, 62)
COLOR_ORANGE_1 = (245, 121, 0)
COLOR_ORANGE_2 = (209, 92, 0)
COLOR_CHOCOLATE_0 = (233, 185, 110)
COLOR_CHOCOLATE_1 = (193, 125, 17)
COLOR_CHOCOLATE_2 = (143, 89, 2)
COLOR_CHAMELEON_0 = (138, 226, 52)
COLOR_CHAMELEON_1 = (115, 210, 22)
COLOR_CHAMELEON_2 = (78, 154, 6)
COLOR_SKY_BLUE_0 = (114, 159, 207)
COLOR_SKY_BLUE_1 = (52, 101, 164)
COLOR_SKY_BLUE_2 = (32, 74, 135)
COLOR_PLUM_0 = (173, 127, 168)
COLOR_PLUM_1 = (117, 80, 123)
COLOR_PLUM_2 = (92, 53, 102)
COLOR_SCARLET_RED_0 = (239, 41, 41)
COLOR_SCARLET_RED_1 = (204, 0, 0)
COLOR_SCARLET_RED_2 = (164, 0, 0)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (211, 215, 207)
COLOR_ALUMINIUM_2 = (186, 189, 182)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_4 = (85, 87, 83)
COLOR_ALUMINIUM_4_5 = (66, 62, 64)
COLOR_ALUMINIUM_5 = (46, 52, 54)


class VisWaymo:
    def __init__(
        self,
        map_valid: np.ndarray,
        map_type: np.ndarray,
        map_pos: np.ndarray,
        map_boundary: np.ndarray,
        px_per_m: float = 10.0,
        video_size: int = 960,
    ) -> None:
        # centered around ego vehicle first step, x=0, y=0, theta=0
        self.px_per_m = px_per_m
        self.video_size = video_size
        self.px_agent2bottom = video_size // 2

        # waymo
        self.lane_style = [
            (COLOR_WHITE, 6),  # FREEWAY = 0
            (COLOR_ALUMINIUM_4_5, 6),  # SURFACE_STREET = 1
            (COLOR_ORANGE_2, 6),  # STOP_SIGN = 2
            (COLOR_CHOCOLATE_2, 6),  # BIKE_LANE = 3
            (COLOR_SKY_BLUE_2, 4),  # TYPE_ROAD_EDGE_BOUNDARY = 4
            (COLOR_PLUM_2, 4),  # TYPE_ROAD_EDGE_MEDIAN = 5
            (COLOR_BUTTER_0, 2),  # BROKEN = 6
            (COLOR_MAGENTA, 2),  # SOLID_SINGLE = 7
            (COLOR_SCARLET_RED_2, 2),  # DOUBLE = 8
            (COLOR_CHAMELEON_2, 4),  # SPEED_BUMP = 9
            (COLOR_SKY_BLUE_0, 4),  # CROSSWALK = 10
        ]

        self.tl_style = [
            COLOR_ALUMINIUM_1,  # STATE_UNKNOWN = 0;
            COLOR_RED,  # STOP = 1;
            COLOR_YELLOW,  # CAUTION = 2;
            COLOR_GREEN,  # GO = 3;
            COLOR_VIOLET,  # FLASHING = 4;
        ]
        # sdc=0, interest=1, predict=2
        self.agent_role_style = [COLOR_CYAN, COLOR_CHAMELEON_2, COLOR_MAGENTA]

        self.agent_cmd_txt = [
            "STATIONARY",  # STATIONARY = 0;
            "STRAIGHT",  # STRAIGHT = 1;
            "STRAIGHT_LEFT",  # STRAIGHT_LEFT = 2;
            "STRAIGHT_RIGHT",  # STRAIGHT_RIGHT = 3;
            "LEFT_U_TURN",  # LEFT_U_TURN = 4;
            "LEFT_TURN",  # LEFT_TURN = 5;
            "RIGHT_U_TURN",  # RIGHT_U_TURN = 6;
            "RIGHT_TURN",  # RIGHT_TURN = 7;
        ]

        raster_map, self.top_left_px = self._register_map(map_boundary, self.px_per_m)
        self.raster_map = self._draw_map(raster_map, map_valid, map_type, map_pos)

    @staticmethod
    def _register_map(map_boundary: np.ndarray, px_per_m: float, edge_px: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            map_boundary: [4], xmin, xmax, ymin, ymax
            px_per_m: float

        Returns:
            raster_map: empty image
            top_left_px
        """
        # y axis is inverted in pixel coordinate
        xmin, xmax, ymax, ymin = (map_boundary * px_per_m).astype(np.int64)
        ymax *= -1
        ymin *= -1
        xmin -= edge_px
        ymin -= edge_px
        xmax += edge_px
        ymax += edge_px

        raster_map = np.zeros([ymax - ymin, xmax - xmin, 3], dtype=np.uint8)
        top_left_px = np.array([xmin, ymin], dtype=np.float32)
        return raster_map, top_left_px

    def _draw_map(
        self,
        raster_map: np.ndarray,
        map_valid: np.ndarray,
        map_type: np.ndarray,
        map_pos: np.ndarray,
        attn_weights_to_pl: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args: numpy arrays
            map_valid: [n_pl, 20],  # bool
            map_type: [n_pl, 11],  # bool one_hot
            map_pos: [n_pl, 20, 2],  # float32
            attn_weights_to_pl: [n_pl], sum up to 1

        Returns:
            raster_map
        """
        mask_valid = map_valid.any(axis=1)
        if attn_weights_to_pl is None:
            attn_weights_to_pl = np.zeros(map_valid.shape[0]) - 1

        for type_to_draw in range(len(self.lane_style)):
            for i in np.where((map_type[:, type_to_draw]) & mask_valid)[0]:
                color, thickness = self.lane_style[type_to_draw]
                if attn_weights_to_pl[i] > 0:
                    color = tuple(np.array(color) * attn_weights_to_pl[i])
                cv2.polylines(
                    raster_map,
                    [self._to_pixel(map_pos[i][map_valid[i]])],
                    isClosed=False,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

        # for i in range(mask_valid.shape[0]):
        #     if mask_valid[i]:
        #         cv2.arrowedLine(
        #             raster_map,
        #             self._to_pixel(map_pos[i, 0]),
        #             self._to_pixel(map_pos[i, 0] + (map_pos[i, 2] - map_pos[i, 0]) * 1),
        #             color=COLOR_CYAN,
        #             thickness=4,
        #             line_type=cv2.LINE_AA,
        #             tipLength=0.5,
        #         )
        return raster_map

    def save_prediction_videos(
        self,
        video_base_name: str,
        episode: Dict[str, np.ndarray],
        prediction: Optional[Dict[str, np.ndarray]],
        save_agent_view: bool = True,
    ) -> List[str]:
        """
        Args:
            episode["agent/valid"]: np.zeros([n_step, N_AGENT], dtype=bool),  # bool,
            episode["agent/pos"]: np.zeros([n_step, N_AGENT, 2], dtype=np.float32),  # x,y
            episode["agent/yaw_bbox"]: np.zeros([n_step, N_AGENT, 1], dtype=np.float32),  # [-pi, pi]
            episode["agent/role"]: np.zeros([N_AGENT, 3], dtype=bool) one_hot [sdc=0, interest=1, predict=2]
            episode["agent/size"]: np.zeros([N_AGENT, 3], dtype=np.float32),  # float32: [length, width, height]
            episode["map/valid"]: np.zeros([N_PL, 20], dtype=bool),  # bool
            episode["map/pos"]: np.zeros([N_PL, 20, 2], dtype=np.float32),  # x,y
            episode["tl_lane/valid"]: np.zeros([n_step, N_TL], dtype=bool),  # bool
            episode["tl_lane/state"]: np.zeros([n_step, N_TL, N_TL_STATE], dtype=bool),  # one_hot
            episode["tl_lane/idx"]: np.zeros([n_step, N_TL], dtype=np.int64) - 1,  # int, -1 means not valid
            episode["tl_stop/valid"]: np.zeros([n_step, N_TL_STOP], dtype=bool),  # bool
            episode["tl_stop/state"]: np.zeros([n_step, N_TL_STOP, N_TL_STATE], dtype=bool),  # one_hot
            episode["tl_stop/pos"]: np.zeros([n_step, N_TL_STOP, 2], dtype=np.float32)  # x,y
            episode["tl_stop/dir"]: np.zeros([n_step, N_TL_STOP, 2], dtype=np.float32)  # x,y
            prediction["step_current"] <= prediction["step_gt"] <= prediction["step_end"]
            prediction["agent/valid"]: [step_current+1...step_end, N_AGENT]
            prediction["agent/pos"]: [step_current+1...step_end, N_AGENT, 2]
            prediction["agent/yaw_bbox"]: [step_current+1...step_end, N_AGENT, 1]
            prediction["speed"]: [step_current+1...step_end, N_AGENT], m/s
            prediction["collided"]: [step_current+1...step_end, N_AGENT], bool
            prediction["collided_this_step"]: [step_current+1...step_end, N_AGENT], bool
            prediction["target_reached"]: [step_current+1...step_end, N_AGENT], bool
            prediction["target_reached_this_step"]: [step_current+1...step_end, N_AGENT], bool
            prediction["route_completed"]: [step_current+1...step_end, N_AGENT], [0,1] percentage
            prediction["route_completed_this_step"]: [step_current+1...step_end, N_AGENT], [0,1] percentage
            prediction["run_red_light"]: [step_current+1...step_end, N_AGENT], bool
            prediction["run_red_light_this_step"]: [step_current+1...step_end, N_AGENT], bool
        """
        buffer_video = {f"{video_base_name}-gt.mp4": [[], None]}  # [List[im], agent_id]
        if prediction is None:
            step_end = episode["agent/valid"].shape[0] - 1
            step_gt = step_end
            step_current = step_end
        else:
            step_current = prediction["step_current"]
            step_end = prediction["step_end"]
            step_gt = prediction["step_gt"]
            buffer_video[f"{video_base_name}-pd.mp4"] = [[], None]
            buffer_video[f"{video_base_name}-mix.mp4"] = [[], None]
            if save_agent_view:
                buffer_video[f"{video_base_name}-sdc.mp4"] = [[], np.where(episode["agent/role"][:, 0])[0][0]]
                for i in np.where(episode["agent/role"][:, 1])[0]:
                    buffer_video[f"{video_base_name}-int_{i}.mp4"] = [[], i]
                for i in np.where(episode["agent/role"][:, 2])[0]:
                    buffer_video[f"{video_base_name}-pre_{i}.mp4"] = [[], i]
                n_others_to_vis = 5
                idx_others = np.where(prediction["agent/valid"].any(0) & ~(episode["agent/role"].any(-1)))[0]
                for i in idx_others[:n_others_to_vis]:
                    buffer_video[f"{video_base_name}-other_{i}.mp4"] = [[], i]

        for t in range(step_end + 1):
            step_image = self.raster_map.copy()
            # draw traffic lights
            if "tl_lane/valid" in episode:
                t_tl = min(t, step_gt)
                for i in range(episode["tl_lane/valid"].shape[1]):
                    if episode["tl_lane/valid"][t_tl, i]:
                        lane_idx = episode["tl_lane/idx"][t_tl, i]
                        tl_state = episode["tl_lane/state"][t_tl, i].argmax()
                        pos = self._to_pixel(episode["map/pos"][lane_idx][episode["map/valid"][lane_idx]])
                        cv2.polylines(
                            step_image,
                            [pos],
                            isClosed=False,
                            color=self.tl_style[tl_state],
                            thickness=8,
                            lineType=cv2.LINE_AA,
                        )
                        if tl_state >= 1 and tl_state <= 3:
                            cv2.drawMarker(
                                step_image,
                                pos[-1],
                                color=self.tl_style[tl_state],
                                markerType=cv2.MARKER_TILTED_CROSS,
                                markerSize=10,
                                thickness=6,
                            )
            # draw traffic lights stop points
            if "tl_stop/valid" in episode:
                for i in range(episode["tl_stop/valid"].shape[1]):
                    if episode["tl_stop/valid"][t_tl, i]:
                        tl_state = episode["tl_stop/state"][t_tl, i].argmax()
                        stop_point = self._to_pixel(episode["tl_stop/pos"][t_tl, i])
                        stop_point_end = self._to_pixel(
                            episode["tl_stop/pos"][t_tl, i] + 5 * episode["tl_stop/dir"][t_tl, i]
                        )
                        cv2.arrowedLine(
                            step_image,
                            stop_point,
                            stop_point_end,
                            color=self.tl_style[tl_state],
                            thickness=4,
                            line_type=cv2.LINE_AA,
                            tipLength=0.3,
                        )

            # draw agents: prediction["step_current"] <= prediction["step_gt"] <= prediction["step_end"]
            step_image_gt = step_image.copy()
            raster_blend_gt = np.zeros_like(step_image)
            if t <= step_gt:
                bbox_gt = self._get_agent_bbox(
                    episode["agent/valid"][t],
                    episode["agent/pos"][t],
                    episode["agent/yaw_bbox"][t],
                    episode["agent/size"],
                )
                bbox_gt = self._to_pixel(bbox_gt)
                agent_role = episode["agent/role"][episode["agent/valid"][t]]
                heading_start = self._to_pixel(episode["agent/pos"][t][episode["agent/valid"][t]])
                heading_end = self._to_pixel(
                    episode["agent/pos"][t][episode["agent/valid"][t]]
                    + 1.5
                    * np.stack(
                        [
                            np.cos(episode["agent/yaw_bbox"][t, :, 0][episode["agent/valid"][t]]),
                            np.sin(episode["agent/yaw_bbox"][t, :, 0][episode["agent/valid"][t]]),
                        ],
                        axis=-1,
                    )
                )
                for i in range(agent_role.shape[0]):
                    if not agent_role[i].any():
                        color = COLOR_ALUMINIUM_0
                    else:
                        color = self.agent_role_style[np.where(agent_role[i])[0].min()]
                    cv2.fillConvexPoly(step_image_gt, bbox_gt[i], color=color)
                    cv2.fillConvexPoly(raster_blend_gt, bbox_gt[i], color=color)
                    cv2.arrowedLine(
                        step_image_gt,
                        heading_start[i],
                        heading_end[i],
                        color=COLOR_BLACK,
                        thickness=4,
                        line_type=cv2.LINE_AA,
                        tipLength=0.6,
                    )
            buffer_video[f"{video_base_name}-gt.mp4"][0].append(step_image_gt)

            if prediction is not None:
                if t > prediction["step_current"]:
                    step_image_pd = step_image.copy()
                    t_pred = t - prediction["step_current"] - 1
                    bbox_pred = self._get_agent_bbox(
                        prediction["agent/valid"][t_pred],
                        prediction["agent/pos"][t_pred],
                        prediction["agent/yaw_bbox"][t_pred],
                        episode["agent/size"],
                    )
                    bbox_pred = self._to_pixel(bbox_pred)
                    heading_start = self._to_pixel(prediction["agent/pos"][t_pred][prediction["agent/valid"][t_pred]])
                    heading_end = self._to_pixel(
                        prediction["agent/pos"][t_pred][prediction["agent/valid"][t_pred]]
                        + 1.5
                        * np.stack(
                            [
                                np.cos(prediction["agent/yaw_bbox"][t_pred, :, 0][prediction["agent/valid"][t_pred]]),
                                np.sin(prediction["agent/yaw_bbox"][t_pred, :, 0][prediction["agent/valid"][t_pred]]),
                            ],
                            axis=-1,
                        )
                    )
                    agent_role = episode["agent/role"][prediction["agent/valid"][t_pred]]
                    for i in range(agent_role.shape[0]):
                        if not agent_role[i].any():
                            color = COLOR_ALUMINIUM_0
                        else:
                            color = self.agent_role_style[np.where(agent_role[i])[0].min()]
                        cv2.fillConvexPoly(step_image_pd, bbox_pred[i], color=color)
                        cv2.arrowedLine(
                            step_image_pd,
                            heading_start[i],
                            heading_end[i],
                            color=COLOR_BLACK,
                            thickness=4,
                            line_type=cv2.LINE_AA,
                            tipLength=0.6,
                        )
                    # step_image_mix = step_image.copy()
                    # cv2.addWeighted(raster_blend_gt, 0.6, step_image_pd, 1, 0, step_image_mix)
                    step_image_mix = cv2.addWeighted(raster_blend_gt, 0.6, step_image_pd, 1, 0)
                else:
                    step_image_pd = step_image_gt.copy()
                    step_image_mix = step_image_gt.copy()
                buffer_video[f"{video_base_name}-pd.mp4"][0].append(step_image_pd)
                buffer_video[f"{video_base_name}-mix.mp4"][0].append(step_image_mix)

            if save_agent_view:
                for k, v in buffer_video.items():
                    agent_idx = v[1]
                    if agent_idx is not None:
                        text_valid = True
                        if t <= step_current:
                            pred_started = False
                            valid_t = t
                            if not episode["agent/valid"][valid_t, agent_idx]:
                                # get the first valid step
                                text_valid = False
                                valid_t = np.where(episode["agent/valid"][:, agent_idx])[0][0]
                            ev_loc = self._to_pixel(episode["agent/pos"][valid_t, agent_idx])
                            ev_rot = episode["agent/yaw_bbox"][valid_t, agent_idx, 0]
                        else:
                            pred_started = True
                            valid_t = t - step_current - 1
                            if not prediction["agent/valid"][valid_t, agent_idx]:
                                # get the closest valid step
                                text_valid = False
                                valid_steps = np.where(prediction["agent/valid"][:, agent_idx])[0]
                                valid_t_idx = np.abs(valid_t - valid_steps).argmin()
                                valid_t = valid_steps[valid_t_idx]
                            ev_loc = self._to_pixel(prediction["agent/pos"][valid_t, agent_idx])
                            ev_rot = prediction["agent/yaw_bbox"][valid_t, agent_idx, 0]

                        agent_view = step_image_mix.copy()
                        if not episode["agent/role"][agent_idx].any():
                            color = COLOR_ALUMINIUM_0
                        else:
                            color = self.agent_role_style[np.where(episode["agent/role"][agent_idx])[0].min()]
                        # draw dest
                        cv2.arrowedLine(
                            agent_view,
                            ev_loc,
                            self._to_pixel(episode["map/pos"][episode["agent/dest"][agent_idx]][0, :2]),
                            color=COLOR_BUTTER_0,
                            thickness=4,
                            line_type=cv2.LINE_AA,
                            tipLength=0.05,
                        )
                        # draw goal
                        cv2.arrowedLine(
                            agent_view,
                            ev_loc,
                            self._to_pixel(episode["agent/goal"][agent_idx, :2]),
                            color=COLOR_MAGENTA,
                            thickness=4,
                            line_type=cv2.LINE_AA,
                            tipLength=0.05,
                        )
                        if "agent/dest" in prediction:
                            cv2.arrowedLine(
                                agent_view,
                                ev_loc,
                                self._to_pixel(episode["map/pos"][prediction["agent/dest"][agent_idx]][0, :2]),
                                color=COLOR_GREEN,
                                thickness=2,
                                line_type=cv2.LINE_AA,
                                tipLength=0.1,
                            )
                        if "agent/goal" in prediction:
                            cv2.arrowedLine(
                                agent_view,
                                ev_loc,
                                self._to_pixel(prediction["agent/goal"][agent_idx, :2]),
                                color=COLOR_GREEN,
                                thickness=2,
                                line_type=cv2.LINE_AA,
                                tipLength=0.1,
                            )
                        trans = self._get_warp_transform(ev_loc, ev_rot)
                        agent_view = cv2.warpAffine(agent_view, trans, (self.video_size, self.video_size))
                        agent_view = self._add_txt(
                            agent_view, episode, prediction, valid_t, agent_idx, pred_started, text_valid
                        )
                        v[0].append(agent_view)

        for k, v in buffer_video.items():
            encoder = ImageEncoder(k, v[0][0].shape, 20, 20)
            for im in v[0]:
                encoder.capture_frame(im)
            encoder.close()
            encoder = None

        return list(buffer_video.keys())

    def save_attn_videos(
        self, video_base_name: str, episode: Dict[str, np.ndarray], prediction: Optional[Dict[str, np.ndarray]]
    ) -> List[str]:
        # todo: deprecated, to be fixed
        step_current = prediction["step_current"]
        step_end = prediction["step_end"]
        step_gt = prediction["step_gt"]

        # normalize attention weights
        # [n_pl, 20]
        valid_pl = episode["map/valid"].any(1)
        # [n_step, n_agent, n_pl]
        attn_min = prediction["attn_weights_to_pl"][:, :, valid_pl].min(axis=(0, 2), keepdims=True)
        attn_max = prediction["attn_weights_to_pl"][:, :, valid_pl].max(axis=(0, 2), keepdims=True)
        attn_weights_to_pl = (prediction["attn_weights_to_pl"] - attn_min) / (attn_max - attn_min + 1e-5)
        # attn_weights_to_pl = prediction["attn_weights_to_pl"] / attn_max

        # [step_current+1...step_end, n_tl]
        n_tl = episode["tl_stop/valid"].shape[-1]
        n_step = prediction["agent/valid"].shape[0]
        valid_tl = np.zeros([n_step, n_tl], dtype=bool)
        valid_tl[: step_gt - step_current] = episode["tl_stop/valid"][step_current + 1 : step_gt + 1]
        valid_tl[step_gt - step_current :] = episode["tl_stop/valid"][step_gt]
        # [n_step, n_agent, n_tl]
        if valid_tl.sum() > 0:
            attn_min = prediction["attn_weights_to_tl"].transpose(0, 2, 1)[valid_tl].min(0)[None, :, None]
            attn_max = prediction["attn_weights_to_tl"].transpose(0, 2, 1)[valid_tl].max(0)[None, :, None]
            attn_weights_to_tl = (prediction["attn_weights_to_tl"] - attn_min) / (attn_max - attn_min + 1e-5)
            # attn_weights_to_tl = prediction["attn_weights_to_tl"] / attn_max
        else:
            # no normalization needed, no valid tl at all.
            attn_weights_to_tl = prediction["attn_weights_to_tl"]

        # [n_step, n_agent, n_agent]
        # attn_min = prediction["attn_weights_to_agent"].transpose(
        # 0, 2, 1)[prediction["agent/valid"]].min(0)[None, :, None]
        # attn_max = prediction["attn_weights_to_agent"].transpose(
        #     0, 2, 1)[prediction["agent/valid"]].max(0)[None, :, None]
        # attn_weights_to_agent = (prediction["attn_weights_to_agent"] - attn_min) / (attn_max - attn_min + 1e-5)
        attn_max = prediction["attn_weights_to_agent"].max(axis=(0, 2), keepdims=True)
        attn_weights_to_agent = prediction["attn_weights_to_agent"] / (attn_max + 1e-5)

        # video_names = {0: f"{video_base_name}-sdc-attn.mp4"} # in case sdc is always the first agent
        video_names = {np.where(episode["agent/role"][:, 0])[0][0]: f"{video_base_name}-sdc-attn.mp4"}

        for i in np.where(episode["agent/role"][:, 1])[0]:
            video_names[i] = f"{video_base_name}-int_{i}-attn.mp4"
        for i in np.where(episode["agent/role"][:, 2])[0]:
            video_names[i] = f"{video_base_name}-pre_{i}-attn.mp4"
        n_others_to_vis = 5
        idx_others = np.where(prediction["agent/valid"].any(0) & ~(episode["agent/role"].any(-1)))[0]
        for i in idx_others[:n_others_to_vis]:
            video_names[i] = f"{video_base_name}-other_{i}-attn.mp4"

        for idx_agent, v_name in video_names.items():
            im_list = []
            for t in range(step_end + 1):
                t_pred = t - step_current - 1
                if t <= step_current:
                    step_image = self.raster_map.copy()
                else:
                    step_image = self._draw_map(
                        raster_map=np.zeros_like(self.raster_map),
                        map_valid=episode["map/valid"],
                        map_type=episode["map/type"],
                        map_pos=episode["map/pos"],
                        attn_weights_to_pl=attn_weights_to_pl[t_pred, idx_agent],
                    )

                # draw agents
                if t <= step_current:
                    agent_bbox = self._get_agent_bbox(
                        episode["agent/valid"][t],
                        episode["agent/pos"][t],
                        episode["agent/yaw_bbox"][t],
                        episode["agent/size"],
                    )
                    agent_bbox = self._to_pixel(agent_bbox)
                    agent_role = np.zeros_like(episode["agent/role"][:, 0])
                    agent_role[idx_agent] = True
                    agent_role = agent_role[episode["agent/valid"][t]]
                    for i in range(agent_role.shape[0]):
                        color = COLOR_CHAMELEON_2 if agent_role[i] else COLOR_ALUMINIUM_0
                        cv2.fillConvexPoly(step_image, agent_bbox[i], color=color)
                else:
                    agent_bbox = self._get_agent_bbox(
                        prediction["agent/valid"][t_pred],
                        prediction["agent/pos"][t_pred],
                        prediction["agent/yaw_bbox"][t_pred],
                        episode["agent/size"],
                    )
                    agent_bbox = self._to_pixel(agent_bbox)
                    agent_role = np.zeros_like(episode["agent/role"][:, 0])
                    agent_role[idx_agent] = True
                    agent_role = agent_role[prediction["agent/valid"][t_pred]]
                    attn_agent = attn_weights_to_agent[t_pred, idx_agent][prediction["agent/valid"][t_pred]]
                    for i in range(agent_role.shape[0]):
                        if agent_role[i]:
                            color = COLOR_CHAMELEON_2
                        else:
                            color = tuple(np.array(COLOR_ALUMINIUM_0) * attn_agent[i])
                        cv2.fillConvexPoly(step_image, agent_bbox[i], color=color)

                # draw traffic lights
                t_tl = min(t, step_gt)
                valid_tl = episode["tl_stop/valid"][t_tl]
                if t <= step_current:
                    for idx_tl in np.where(valid_tl)[0]:
                        tl_state = episode["tl_stop/state"][t_tl, idx_tl].argmax()
                        stop_point = self._to_pixel(episode["tl_stop/pos"][t_tl, idx_tl])
                        cv2.circle(step_image, stop_point, 10, color=self.tl_style[tl_state], thickness=-1)
                else:
                    for idx_tl in np.where(valid_tl)[0]:
                        tl_state = episode["tl_stop/state"][t_tl, idx_tl].argmax()
                        stop_point = self._to_pixel(episode["tl_stop/pos"][t_tl, idx_tl])
                        attn_tl = attn_weights_to_tl[t_pred, idx_agent, idx_tl]
                        color = tuple(np.array(self.tl_style[tl_state]) * attn_tl)
                        cv2.circle(step_image, stop_point, 10, color=color, thickness=-1)

                im_list.append(step_image)

            encoder = ImageEncoder(v_name, im_list[0].shape, 20, 20)
            for im in im_list:
                encoder.capture_frame(im)
            encoder.close()
            encoder = None

        return [v_name for _, v_name in video_names.items()]

    @staticmethod
    def _add_txt(
        im: np.ndarray,
        episode: Dict[str, np.ndarray],
        prediction: Optional[Dict[str, np.ndarray]],
        t: int,
        idx: int,
        pred_started: bool,
        text_valid: bool,
        line_width: int = 30,
    ) -> np.ndarray:
        h, w, _ = im.shape
        agent_view = np.zeros([h, w + 200, 3], dtype=im.dtype)
        agent_view[:h, :w] = im
        if (prediction is not None) and pred_started:
            txt_list = [
                f'id:{int(episode["episode_idx"])}',
                f"valid:{int(text_valid)}",
                f'goal_valid:{int(prediction["goal_valid"][t, idx])}',
                f'out:{int(prediction["outside_map_this_step"][t, idx])}/{int(prediction["outside_map"][t, idx])}',
                f'col:{int(prediction["collided_this_step"][t, idx])}/{int(prediction["collided"][t, idx])}',
                f'red:{int(prediction["run_red_light_this_step"][t, idx])}/{int(prediction["run_red_light"][t, idx])}',
                f'edge:{int(prediction["run_road_edge_this_step"][t, idx])}/{int(prediction["run_road_edge"][t, idx])}',
                f'passive:{int(prediction["passive_this_step"][t, idx])}/{int(prediction["passive"][t, idx])}',
                f'r_goal:{int(prediction["goal_reached_this_step"][t, idx])}/{int(prediction["goal_reached"][t, idx])}',
                f'r_dest:{int(prediction["dest_reached_this_step"][t, idx])}/{int(prediction["dest_reached"][t, idx])}',
                f'x:{prediction["agent/pos"][t, idx, 0]:.2f}',
                f'y:{prediction["agent/pos"][t, idx, 1]:.2f}',
                f'spd:{prediction["speed"][t, idx]:.2f}',
                f'yaw:{prediction["agent/yaw_bbox"][t, idx, 0]:.2f}',
                f'acc:{prediction["action"][t, idx, 0]:.2f}',
                f'steer:{prediction["action"][t, idx, 1]:.2f}',
                f'act_P:{prediction["act_P"][t, idx]:.2f}',
                f'lat_P:{prediction["lat_P"][t, idx]:.2f}',
            ]
            if "score" in prediction:
                txt_list.append(f'score:{prediction["score"][idx]:.2f}')
            for k in [
                "rew_pg",
                "val_pg",
                "pb_pg",
                "adv_pg",
                "ret_pg",
                "rew_dr",
                "val_dr",
                "val_tgt",
                "ret_dr",
                "ret_tgt",
            ]:
                if k in prediction:
                    txt_list.append(f"{k}:{prediction[k][t, idx]:.2f}")
        else:
            txt_list = [
                f'id:{int(episode["episode_idx"])}',
                f"valid:{int(text_valid)}",
                f'g_x:{episode["agent/goal"][idx, 0]:.2f}',
                f'g_y:{episode["agent/goal"][idx, 1]:.2f}',
                f'x:{episode["agent/pos"][t, idx, 0]:.2f}',
                f'y:{episode["agent/pos"][t, idx, 1]:.2f}',
                f'yaw:{episode["agent/yaw_bbox"][t, idx, 0]:.2f}',
                f'spd:{episode["agent/spd"][t, idx, 0]:.2f}',
                f'role:{list(np.where(episode["agent/role"][idx])[0])}',
                f'size_x:{episode["agent/size"][idx, 0]:.2f}',
                f'size_y:{episode["agent/size"][idx, 1]:.2f}',
                f'size_z:{episode["agent/size"][idx, 2]:.2f}',
            ]

        for i, txt in enumerate(txt_list):
            agent_view = cv2.putText(
                agent_view, txt, (w, line_width * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
            )
        return agent_view

    def _to_pixel(self, pos: np.ndarray) -> np.ndarray:
        pos = pos * self.px_per_m
        pos[..., 0] = pos[..., 0] - self.top_left_px[0]
        pos[..., 1] = -pos[..., 1] - self.top_left_px[1]
        return np.round(pos).astype(np.int32)

    def _get_warp_transform(self, loc: np.ndarray, yaw: float) -> np.ndarray:
        """
        loc: xy in pixel
        yaw: in rad
        """

        forward_vec = np.array([np.cos(yaw), -np.sin(yaw)])
        right_vec = np.array([np.sin(yaw), np.cos(yaw)])

        bottom_left = loc - self.px_agent2bottom * forward_vec - (0.5 * self.video_size) * right_vec
        top_left = loc + (self.video_size - self.px_agent2bottom) * forward_vec - (0.5 * self.video_size) * right_vec
        top_right = loc + (self.video_size - self.px_agent2bottom) * forward_vec + (0.5 * self.video_size) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self.video_size - 1], [0, 0], [self.video_size - 1, 0]], dtype=np.float32)
        return cv2.getAffineTransform(src_pts, dst_pts)

    @staticmethod
    def _normalize_attention(attn: np.ndarray, valid: np.ndarray) -> np.ndarray:
        # attn = attn / attn.max()
        attn_min = attn[valid].min()
        attn_max = attn[valid].max()
        attn = (attn - attn_min) / (attn_max - attn_min + 1e-5)
        return attn

    @staticmethod
    def _get_agent_bbox(
        agent_valid: np.ndarray, agent_pos: np.ndarray, agent_yaw: np.ndarray, agent_size: np.ndarray
    ) -> np.ndarray:
        yaw = agent_yaw[agent_valid]  # n, 1
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        v_forward = np.concatenate([cos_yaw, sin_yaw], axis=-1)  # n,2
        v_right = np.concatenate([sin_yaw, -cos_yaw], axis=-1)

        offset_forward = 0.5 * agent_size[agent_valid, 0:1] * v_forward  # [n, 2]
        offset_right = 0.5 * agent_size[agent_valid, 1:2] * v_right  # [n, 2]

        vertex_offset = np.stack(
            [
                -offset_forward + offset_right,
                offset_forward + offset_right,
                offset_forward - offset_right,
                -offset_forward - offset_right,
            ],
            axis=1,
        )  # n,4,2

        agent_pos = agent_pos[agent_valid]
        bbox = agent_pos[:, None, :].repeat(4, 1) + vertex_offset  # n,4,2
        return bbox

    def get_dest_prob_image(
        self,
        im_base_name: str,
        episode: Dict[str, np.ndarray],
        prediction: Dict[str, np.ndarray],
        dest_prob: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            episode["agent/valid"]: np.zeros([n_step, N_AGENT], dtype=bool),  # bool,
            episode["agent/pos"]: np.zeros([n_step, N_AGENT, 2], dtype=np.float32),  # x,y
            episode["agent/yaw_bbox"]: np.zeros([n_step, N_AGENT, 1], dtype=np.float32),  # [-pi, pi]
            episode["agent/role"]: np.zeros([N_AGENT, 3], dtype=bool) one_hot [sdc=0, interest=1, predict=2]
            episode["agent/size"]: np.zeros([N_AGENT, 3], dtype=np.float32),  # float32: [length, width, height]
            dest_prob: [n_agent, n_pl] float prob
        """
        list_im_idx = {}
        list_im_idx[f"{im_base_name}-sdc.jpg"] = 0
        for i in np.where(episode["agent/role"][:, 1])[0]:
            list_im_idx[f"{im_base_name}-int_{i}.jpg"] = i
        for i in np.where(episode["agent/role"][:, 2])[0]:
            list_im_idx[f"{im_base_name}-pre_{i}.jpg"] = i
        n_others_to_vis = 5
        idx_others = np.where(prediction["agent/valid"].any(0) & ~(episode["agent/role"].any(-1)))[0]
        for i in idx_others[:n_others_to_vis]:
            list_im_idx[f"{im_base_name}-other_{i}.jpg"] = i

        for im_path, i in list_im_idx.items():
            t = episode["agent/valid"][:, i].argmax()
            dest_valid = dest_prob[i] > 1e-4
            p_normalized = dest_prob[i, dest_valid].copy()
            p_max = p_normalized.max()
            p_min = p_normalized.min()
            p_normalized = (p_normalized - p_min) / (p_max - p_min + 1e-4) * 3.0
            m_type = np.zeros_like(episode["map/type"][dest_valid])
            m_type[:, 1] = True
            for k in p_normalized.argsort()[-6:]:
                m_type[k, 1] = False
                m_type[k, 3] = True
            im = self._draw_map(
                raster_map=np.zeros_like(self.raster_map),
                map_valid=episode["map/valid"][dest_valid],
                map_type=m_type,
                map_pos=episode["map/pos"][dest_valid],
                attn_weights_to_pl=p_normalized,
            )

            draw_gt_dest = self._to_pixel(episode["map/pos"][episode["agent/dest"][i]])
            # [n_pl, 20]
            draw_gt_dest_valid = episode["map/valid"][episode["agent/dest"][i]]
            cv2.polylines(
                im,
                [draw_gt_dest[draw_gt_dest_valid]],
                isClosed=False,
                color=COLOR_MAGENTA,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            bbox_gt = self._get_agent_bbox(
                episode["agent/valid"][t, i],
                episode["agent/pos"][t, i],
                episode["agent/yaw_bbox"][t, i],
                episode["agent/size"][i],
            )
            bbox_gt = self._to_pixel(bbox_gt)
            heading_start = self._to_pixel(episode["agent/pos"][t, i])
            heading_end = self._to_pixel(
                episode["agent/pos"][t, i]
                + 1.5
                * np.stack(
                    [np.cos(episode["agent/yaw_bbox"][t, i, 0]), np.sin(episode["agent/yaw_bbox"][t, i, 0])], axis=-1
                )
            )
            cv2.fillConvexPoly(im, bbox_gt, color=COLOR_RED)
            cv2.arrowedLine(
                im, heading_start, heading_end, color=COLOR_BLACK, thickness=4, line_type=cv2.LINE_AA, tipLength=0.6
            )
            cv2.imwrite(im_path, im[..., ::-1])
        return list(list_im_idx.keys())
