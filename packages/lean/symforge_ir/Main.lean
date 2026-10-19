import SymforgeIr.Types
import SymforgeIr.Properties
import Lean.Data.Json

open SymforgeIr
open Lean

def main (args : List String) : IO UInt32 := do
  if args.isEmpty then
    IO.println "SymForge Lean 4 IR CLI"
    IO.println "Usage: symforge_cli [check | example-audio | example-sdf]"
    return 0

  match args.head! with
  | "example-audio" =>
    let json := toJson exampleAudioPatch
    IO.println (json.compress)
    return 0
  | "example-sdf" =>
    let json := toJson exampleSDFPatch
    IO.println (json.compress)
    return 0
  | "verify" =>
    IO.println s!"Verified Audio Patch size: {exampleAudioPatch.size}, depth: {exampleAudioPatch.depth}"
    IO.println s!"Verified SDF Patch size: {exampleSDFPatch.size}, depth: {exampleSDFPatch.depth}"
    return 0
  | "parse-audio" =>
    let stdin ← IO.getStdin
    let input ← stdin.readToEnd
    match Json.parse input with
    | Except.error err =>
      IO.eprintln s!"JSON parse error: {err}"
      return 1
    | Except.ok json =>
      match (fromJson? json : Except String AudioNode) with
      | Except.error err =>
        IO.eprintln s!"AudioNode decode error: {err}"
        return 2
      | Except.ok node =>
        let res : Json := Json.mkObj [
          ("status", Json.str "ok"),
          ("size", Json.num node.size),
          ("depth", Json.num node.depth),
          ("paramCount", Json.num node.collectParams.length)
        ]
        IO.println res.compress
        return 0
  | "parse-sdf" =>
    let stdin ← IO.getStdin
    let input ← stdin.readToEnd
    match Json.parse input with
    | Except.error err =>
      IO.eprintln s!"JSON parse error: {err}"
      return 1
    | Except.ok json =>
      match (fromJson? json : Except String SDFNode) with
      | Except.error err =>
        IO.eprintln s!"SDFNode decode error: {err}"
        return 2
      | Except.ok node =>
        let res : Json := Json.mkObj [
          ("status", Json.str "ok"),
          ("size", Json.num node.size),
          ("depth", Json.num node.depth),
          ("paramCount", Json.num node.collectParams.length)
        ]
        IO.println res.compress
        return 0
  | other =>
    IO.eprintln s!"Unknown command: {other}"
    return 1
