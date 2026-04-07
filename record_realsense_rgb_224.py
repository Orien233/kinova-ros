#!/usr/bin/env python3


# -*- coding: utf-8 -*-




import os


import csv


import time


import signal


import argparse


from pathlib import Path




import cv2


import numpy as np


import pyrealsense2 as rs






class RGBRecorder:


    def __init__(


        self,


        raw_dir: str,


        fused_dir: str,


        width: int = 640,


        height: int = 480,


        stream_fps: int = 30,


        save_hz: float = 10.0,


        output_size: int = 224,


        center_ratio: float = 1.0,


        image_ext: str = "png",


        jpeg_quality: int = 95,


        preview: bool = False,


    ):


        self.raw_dir = Path(raw_dir)


        self.fused_dir = Path(fused_dir)


        self.width = width


        self.height = height


        self.stream_fps = stream_fps


        self.save_hz = save_hz


        self.output_size = output_size


        self.center_ratio = center_ratio


        self.image_ext = image_ext.lower()


        self.jpeg_quality = jpeg_quality


        self.preview = preview




        self.running = True


        self.pipeline = None


        self.config = None




        self.save_interval = 1.0 / self.save_hz


        self.next_save_time = None


        self.frame_index = 0




        self.raw_dir.mkdir(parents=True, exist_ok=True)


        self.fused_dir.mkdir(parents=True, exist_ok=True)




        self.meta_path = self.raw_dir.parent / "metadata.csv"




    def _build_pipeline(self):


        self.pipeline = rs.pipeline()


        self.config = rs.config()




        # 仅启用 RGB


        self.config.enable_stream(


            rs.stream.color,


            self.width,


            self.height,


            rs.format.bgr8,


            self.stream_fps


        )




        profile = self.pipeline.start(self.config)


        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()


        intr = color_stream.get_intrinsics()




        print("[INFO] RealSense RGB started")


        print(f"[INFO] Resolution: {intr.width}x{intr.height}")


        print(f"[INFO] fx={intr.fx:.3f}, fy={intr.fy:.3f}, ppx={intr.ppx:.3f}, ppy={intr.ppy:.3f}")




    def _stop_pipeline(self):


        if self.pipeline is not None:


            try:


                self.pipeline.stop()


            except Exception:


                pass




    def _center_crop_square(self, image: np.ndarray) -> np.ndarray:


        """


        先从原图中心裁出一个正方形区域。


        center_ratio=1.0 表示取最大中心正方形；


        center_ratio<1.0 表示只取更小的中心区域。


        """


        h, w = image.shape[:2]


        side = min(h, w)


        crop_side = int(side * self.center_ratio)




        # 保证最终裁剪边长至少 >= output_size


        crop_side = max(crop_side, self.output_size)




        cx = w // 2


        cy = h // 2


        half = crop_side // 2




        x1 = max(cx - half, 0)


        y1 = max(cy - half, 0)


        x2 = x1 + crop_side


        y2 = y1 + crop_side




        # 边界修正


        if x2 > w:


            x2 = w


            x1 = w - crop_side


        if y2 > h:


            y2 = h


            y1 = h - crop_side




        return image[y1:y2, x1:x2]




    def _make_fused_224(self, bgr_image: np.ndarray) -> np.ndarray:


        """


        中心裁剪 + 面积插值缩小到 224x224。


        INTER_AREA 适合图像缩小场景，可理解为一种像素面积融合/重采样。


        """


        cropped = self._center_crop_square(bgr_image)


        fused = cv2.resize(


            cropped,


            (self.output_size, self.output_size),


            interpolation=cv2.INTER_AREA


        )


        return fused




    def _encode_params(self):


        if self.image_ext in ["jpg", "jpeg"]:


            return [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]


        return []




    def _save_pair(self, rgb_bgr: np.ndarray, fused_bgr: np.ndarray, timestamp_wall: float):


        stem = f"{self.frame_index:06d}_{int(timestamp_wall * 1000)}"


        raw_path = self.raw_dir / f"{stem}.{self.image_ext}"


        fused_path = self.fused_dir / f"{stem}.{self.image_ext}"




        ok1 = cv2.imwrite(str(raw_path), rgb_bgr, self._encode_params())


        ok2 = cv2.imwrite(str(fused_path), fused_bgr, self._encode_params())




        if not ok1 or not ok2:


            raise RuntimeError(f"Failed to save images: {raw_path}, {fused_path}")




        return raw_path.name, fused_path.name




    def _init_metadata(self):


        file_exists = self.meta_path.exists()


        self.meta_fp = open(self.meta_path, "a", newline="", encoding="utf-8")


        self.meta_writer = csv.writer(self.meta_fp)




        if not file_exists:


            self.meta_writer.writerow([


                "frame_index",


                "timestamp_wall_sec",


                "raw_filename",


                "fused_filename",


                "raw_width",


                "raw_height",


                "fused_width",


                "fused_height",


                "center_ratio",


                "save_hz"


            ])


            self.meta_fp.flush()




    def _close_metadata(self):


        if hasattr(self, "meta_fp") and self.meta_fp:


            self.meta_fp.close()




    def stop(self, *_):


        print("\n[INFO] Stop signal received.")


        self.running = False




    def run(self):


        signal.signal(signal.SIGINT, self.stop)


        signal.signal(signal.SIGTERM, self.stop)




        self._build_pipeline()


        self._init_metadata()




        # 给相机一点预热时间


        for _ in range(10):


            self.pipeline.wait_for_frames()




        self.next_save_time = time.monotonic()




        try:


            while self.running:


                frames = self.pipeline.wait_for_frames()


                color_frame = frames.get_color_frame()


                if not color_frame:


                    continue




                now_mono = time.monotonic()


                if now_mono < self.next_save_time:


                    if self.preview:


                        bgr = np.asanyarray(color_frame.get_data())


                        fused = self._make_fused_224(bgr)


                        cv2.imshow("raw_rgb", bgr)


                        cv2.imshow("fused_224", fused)


                        key = cv2.waitKey(1) & 0xFF


                        if key == 27:  # ESC


                            self.running = False


                    continue




                # 到保存时间了：取当前帧


                bgr = np.asanyarray(color_frame.get_data())


                fused = self._make_fused_224(bgr)




                timestamp_wall = time.time()


                raw_name, fused_name = self._save_pair(bgr, fused, timestamp_wall)




                self.meta_writer.writerow([


                    self.frame_index,


                    f"{timestamp_wall:.6f}",


                    raw_name,


                    fused_name,


                    bgr.shape[1],


                    bgr.shape[0],


                    fused.shape[1],


                    fused.shape[0],


                    self.center_ratio,


                    self.save_hz


                ])


                self.meta_fp.flush()




                print(f"[SAVE] idx={self.frame_index:06d} raw={raw_name} fused={fused_name}")




                self.frame_index += 1




                # 防止漂移：按固定周期推进


                while self.next_save_time <= now_mono:


                    self.next_save_time += self.save_interval




                if self.preview:


                    cv2.imshow("raw_rgb", bgr)


                    cv2.imshow("fused_224", fused)


                    key = cv2.waitKey(1) & 0xFF


                    if key == 27:


                        self.running = False




        finally:


            self._stop_pipeline()


            self._close_metadata()


            if self.preview:


                cv2.destroyAllWindows()


            print("[INFO] Recorder exited cleanly.")






def parse_args():


    parser = argparse.ArgumentParser(


        description="Record RealSense D435 RGB images at 10Hz and save center-fused 224x224 images."


    )


    parser.add_argument("--raw-dir", type=str, default="./output/rgb_raw",


                        help="Directory to save raw RGB images")


    parser.add_argument("--fused-dir", type=str, default="./output/rgb_224",


                        help="Directory to save fused 224x224 RGB images")


    parser.add_argument("--width", type=int, default=640,


                        help="Color stream width")


    parser.add_argument("--height", type=int, default=480,


                        help="Color stream height")


    parser.add_argument("--stream-fps", type=int, default=30,


                        help="RealSense RGB streaming FPS")


    parser.add_argument("--save-hz", type=float, default=10.0,


                        help="Image saving frequency")


    parser.add_argument("--output-size", type=int, default=224,


                        help="Output fused image size")


    parser.add_argument("--center-ratio", type=float, default=1.0,


                        help="Central square crop ratio relative to min(H,W). 1.0 means maximum central square")


    parser.add_argument("--ext", type=str, default="png", choices=["png", "jpg", "jpeg"],


                        help="Image file extension")


    parser.add_argument("--jpeg-quality", type=int, default=95,


                        help="JPEG quality if ext=jpg/jpeg")


    parser.add_argument("--preview", action="store_true",


                        help="Show preview windows")


    return parser.parse_args()






def main():


    args = parse_args()




    recorder = RGBRecorder(


        raw_dir=args.raw_dir,


        fused_dir=args.fused_dir,


        width=args.width,


        height=args.height,


        stream_fps=args.stream_fps,


        save_hz=args.save_hz,


        output_size=args.output_size,


        center_ratio=args.center_ratio,


        image_ext=args.ext,


        jpeg_quality=args.jpeg_quality,


        preview=args.preview,


    )


    recorder.run()






if __name__ == "__main__":


    main()



