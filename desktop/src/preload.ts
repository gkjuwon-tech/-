import { contextBridge, ipcRenderer } from "electron";

/**
 * 렌더러 ↔ 메인 안전 브리지. 위젯은 단일 'state' 채널만 받는다(일관 포즈 PNG).
 */
contextBridge.exposeInMainWorld("kkoji", {
  // 메인 → 렌더러
  onState: (cb: (data: unknown) => void) =>
    ipcRenderer.on("state", (_e, data) => cb(data)),
  onMode: (cb: (data: unknown) => void) =>
    ipcRenderer.on("mode", (_e, data) => cb(data)),
  onAutotype: (cb: (data: unknown) => void) =>
    ipcRenderer.on("autotype", (_e, data) => cb(data)),

  // 렌더러 → 메인
  ready: () => ipcRenderer.send("ready"),
  pet: () => ipcRenderer.send("pet"),
  menu: () => ipcRenderer.send("menu"),
  goneClick: () => ipcRenderer.send("gone-click"),
  talk: (text: string) => ipcRenderer.send("talk", text),

  // 프롬프트(키 입력) 창
  onPromptInit: (cb: (data: unknown) => void) =>
    ipcRenderer.on("prompt:init", (_e, data) => cb(data)),
  promptSubmit: (value: string | null) =>
    ipcRenderer.send("prompt:submit", value),
});
