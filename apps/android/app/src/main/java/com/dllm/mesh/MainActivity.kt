package com.dllm.mesh

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Devices
import androidx.compose.material.icons.filled.PieChart
import androidx.compose.material.icons.filled.QrCodeScanner
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.dllm.mesh.ui.ChatScreen
import com.dllm.mesh.ui.DevicesScreen
import com.dllm.mesh.ui.DllmMeshTheme
import com.dllm.mesh.ui.MeshColors
import com.dllm.mesh.ui.PairingScreen
import com.dllm.mesh.ui.SettingsScreen
import com.dllm.mesh.ui.UsageScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DllmMeshTheme {
                DllmMeshApp()
            }
        }
    }
}

/**
 * One bottom-nav tab. The route doubles as the nav graph destination, so a tab
 * can never point at a screen that does not exist.
 */
private data class MeshTab(
    val route: String,
    val label: String,
    val icon: ImageVector,
)

/**
 * Tab order: chat first (it is where the work happens), then the mesh itself,
 * then the two ways to change it (pair a coordinator, see what it cost), and
 * settings last. The old Networks tab is gone — the mesh is a single network, so
 * "Devices" is the only roster there is.
 */
private val TABS = listOf(
    MeshTab("chat", "Chat", Icons.Filled.Chat),
    MeshTab("devices", "Devices", Icons.Filled.Devices),
    MeshTab("pairing", "Pairing", Icons.Filled.QrCodeScanner),
    MeshTab("usage", "Usage", Icons.Filled.PieChart),
    MeshTab("settings", "Settings", Icons.Filled.Settings),
)

private const val START_ROUTE = "chat"

@Composable
fun DllmMeshApp() {
    val navController = rememberNavController()
    val backStack by navController.currentBackStackEntryAsState()
    val route = backStack?.destination?.route ?: START_ROUTE

    Scaffold(
        containerColor = MeshColors.Ink,
        bottomBar = {
            NavigationBar(containerColor = MeshColors.Panel) {
                TABS.forEach { tab ->
                    NavigationBarItem(
                        selected = route == tab.route,
                        onClick = {
                            if (route == tab.route) return@NavigationBarItem
                            // popUpTo(start) + saveState keeps ONE entry per tab
                            // instead of appending a new copy on every tap, and
                            // restoreState puts each tab back where the user left
                            // it. Without this the back stack grew without bound
                            // and Back walked through every tab visit in reverse
                            // rather than stepping back through the tabs.
                            navController.navigate(tab.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = { Icon(tab.icon, contentDescription = null) },
                        label = { Text(tab.label) },
                    )
                }
            }
        },
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = START_ROUTE,
            modifier = Modifier.padding(padding),
        ) {
            composable("chat") { ChatScreen() }
            composable("devices") { DevicesScreen() }
            composable("pairing") { PairingScreen() }
            composable("usage") { UsageScreen() }
            composable("settings") { SettingsScreen() }
        }
    }
}
