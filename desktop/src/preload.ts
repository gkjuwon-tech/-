import { contextBridge, ipcRenderer } from "electron";

/**
 * 렌더러 ↔ 메인 안전 브리지. contextIsolation 켠 상태에서 필요한 채널만 노출.
 */
contextBridge.exposeInMainWorld("kkoji", {
  // 캐릭터 창
  onState: (cb: (data: unknown) => void) =>
    ipcRenderer.on("state", (_e, data) => cb(data)),
  onSay: (cb: (data: unknown) => void) =>
    ipcRenderer.on("say", (_e, data) => cb(data)),
  pet: () => ipcRenderer.send("pet"),
  menu: () => ipcRenderer.send("menu"),

  // 프롬프트(키 입력) 창
  onPromptInit: (cb: (data: unknown) => void) =>
    ipcRenderer.on("prompt:init", (_e, data) => cb(data)),
  promptSubmit: (value: string | null) =>
    ipcRenderer.send("prompt:submit", value),

  // 엔딩 창
  onEndingInit: (cb: (data: unknown) => void) =>
    ipcRenderer.on("ending:init", (_e, data) => cb(data)),
  endingDone: () => ipcRenderer.send("ending:done"),
});
