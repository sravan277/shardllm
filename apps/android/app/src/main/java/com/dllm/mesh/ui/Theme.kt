package com.dllm.mesh.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/**
 * The single source of truth for the app's colour palette.
 *
 * WHY an object instead of `MaterialTheme.colorScheme`: this app talks about
 * LAN-signal state (is this device live? is the fingerprint pinned? is the load
 * real?) and those meanings must not drift with the Material scheme. Every
 * screen reads the same constants here, so "teal = good/live, amber = warning"
 * is a project-wide contract instead of a per-file copy that silently diverges.
 *
 * WHY these values: an operator glances at a phone screen across a room. Ink /
 * panel are near-black so the UI reads as one dark surface, teal is the single
 * accent for live/connected state, amber is reserved exclusively for warnings
 * (TOFU mismatch, unreachable host, stale data) and muted grey for anything the
 * server has not actually reported.
 */
object MeshColors {
    /** App background / darkest surface. */
    val Ink = Color(0xFF101418)

    /** Resting card surface (list rows, panels). */
    val Panel = Color(0xFF171D24)

    /** Accent surface for the selected / active row. */
    val PanelActive = Color(0xFF1B2B28)

    /** Primary accent: connected, live, matched, running. */
    val Teal = Color(0xFF2DD4BF)

    /** Warning accent only: mismatch, unreachable, stale, failed action. */
    val Amber = Color(0xFFF5B544)

    /** Body text on dark surfaces. */
    val Text = Color(0xFFE8EDF2)

    /** Secondary text, and the colour for "not reported" state. */
    val Muted = Color(0xFF93A1B0)
}

/** LAN-signal theme: ink panels, single teal accent, amber reserved for warnings. */
@Composable
fun DllmMeshTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = MeshColors.Ink,
            surface = MeshColors.Panel,
            primary = MeshColors.Teal,
            secondary = MeshColors.Muted,
            tertiary = MeshColors.Amber,
            onBackground = MeshColors.Text,
            onSurface = MeshColors.Text,
        ),
        content = content,
    )
}
