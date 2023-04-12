using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using MahApps.Metro.Controls;
using ControlzEx.Theming;
using CQ.Enterprise.App.AlgoPlatform.Applications;

namespace CQ.Enterprise.App.AlgoPlatform
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : MetroWindow
    {
        public MainWindow()
        {
            InitializeComponent();

            ThemeManager.Current.ChangeTheme(this, "Light.Blue");
        }

        private void button_BigDataMgr_Clicked(object sender, RoutedEventArgs e)
        {
            StraSugWindow win1 = new StraSugWindow();
            win1.Show();
        }


    }
}
